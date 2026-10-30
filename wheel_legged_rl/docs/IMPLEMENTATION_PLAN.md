# Go2-W 四足四輪 RL 環境 — 調査結果と実装計画

> この文書は実装の元になった計画書（`go2w-wheel-leg-rl-plan.md`）の写しです。実装時の判断・計画からの差分・検証結果は [DEV_LOG.md](DEV_LOG.md) に記録しています。

調査日: 2026-09-23
調査対象（すべて実物を clone して確認）:

| 対象 | バージョン |
|---|---|
| `unitreerobotics/unitree_rl_lab` | main `4960b84` (2025-11-19)。README 表記: Isaac Sim 5.1.0 / Isaac Lab 2.3.0 |
| `unitreerobotics/unitree_ros` → `robots/go2w_description/urdf/go2w_description.urdf` | master |
| `isaac-sim/IsaacLab` | tag `v2.3.0` (2025-10-28)。API 確認はすべてこのソースに対して実施 |
| `rsl-rl-lib` | 3.0.1（`isaaclab_rl/setup.py` の固定バージョン） |

> 本計画の数値のうち **URDF / リポジトリから確認した事実** と **設計上の選択（推定値を含む）** は区別して書いています。推定値には「推定」と明記しました。

---

## 1. unitree_rl_lab のディレクトリ構造

```text
unitree_rl_lab/
├── scripts/
│   ├── list_envs.py
│   ├── rsl_rl/{train.py, play.py, cli_args.py}   # Isaac Lab の rsl_rl スクリプトとほぼ同じ + deploy.yaml 出力
│   └── mimic/...
├── source/unitree_rl_lab/unitree_rl_lab/
│   ├── assets/robots/
│   │   ├── unitree.py            # UNITREE_GO2_CFG, UNITREE_GO2W_CFG, B2, H1, G1 ... (ArticulationCfg)
│   │   └── unitree_actuators.py  # UnitreeActuator: DelayedPDActuator + T-N カーブ + 摩擦
│   ├── tasks/locomotion/
│   │   ├── agents/rsl_rl_ppo_cfg.py   # BasePPORunnerCfg（全ロボット共通）
│   │   ├── mdp/{commands, curriculums, observations, rewards}.py
│   │   └── robots/{go2, h1, g1/29dof}/velocity_env_cfg.py  # 1 ファイルに Scene/Obs/Act/Rew/...
│   ├── tasks/mimic/...
│   └── utils/{export_deploy_cfg.py, parser_cfg.py}
└── deploy/                        # C++ 実機コントローラ（ONNX Runtime）
    ├── include/isaaclab/...       # Isaac Lab の obs/action manager を C++ で再実装
    └── robots/{go2, go2w, b2, g1_*, h1*}/
```

**重要な発見**

1. `UNITREE_GO2W_CFG`（アセット定義）と `deploy/robots/go2w/`（C++ 実機コントローラ）は存在するが、**Go2-W 用の学習タスク（env cfg）は公開されていない**。
2. `deploy/robots/go2w/src/State_RLBase.cpp` は `processed_actions[0:12]` を関節位置 `q`、`[12:16]` を車輪速度 `dq` として送る。つまり Unitree 自身も **「12 脚位置 + 4 車輪速度」** の 16 次元 Action を前提としている。本計画の Action 定義と一致しており、この C++ コントローラと互換な `deploy.yaml` を出力できるようにする。
3. `UNITREE_GO2W_CFG` は全 16 関節に `IdealPDActuatorCfg(effort_limit=23.5, stiffness 脚25 / 車輪0, damping=0.5)`。calf の実トルク上限（URDF 35.55 Nm）が反映されていない。

## 2. Go2 用 RL 環境の構成（`robots/go2/velocity_env_cfg.py`）

| 要素 | 内容 |
|---|---|
| Env | `isaaclab.envs:ManagerBasedRLEnv`、`decimation=4`, `sim.dt=0.005`（50 Hz 制御）、`episode_length_s=20` |
| Scene | TerrainGenerator（実質 flat のみ有効、他はコメントアウト）、height_scanner（定義のみ・観測未使用）、ContactSensor(`Robot/.*`) |
| Robot / Actuator | `UnitreeActuatorCfg_Go2HV`（Kp 25, Kd 0.5, X1 13.5, X2 30, Y1 20.2, Y2 23.4 のT-Nカーブ） |
| Action | `JointPositionActionCfg(joint_names=[".*"], scale=0.25, use_default_offset=True)` → 12 次元 |
| Policy obs | base_ang_vel(×0.2), projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel(×0.05), last_action = 45 次元 |
| Critic obs | 上記 + base_lin_vel + joint_effort（非対称 Actor-Critic） |
| Reward | track_lin_vel_xy 1.5 / track_ang_vel_z 0.75 / lin_vel_z −2 / ang_vel_xy −0.05 / joint_vel, acc, torques / action_rate −0.1 / joint_pos_limits −10 / energy / flat_orientation −2.5 / joint_position_penalty / feet_air_time / air_time_variance / feet_slide / undesired_contacts |
| Termination | time_out, base contact, bad_orientation(0.8 rad) |
| Curriculum | `terrain_levels_vel`, `lin_vel_cmd_levels`（追従報酬が閾値を超えたらコマンド範囲拡大） |
| DR | 摩擦(0.3–1.2), base 質量(−1〜+3 kg), push, reset 時の関節速度 |
| PPO | rsl_rl OnPolicyRunner, 24 steps/env, actor/critic [512,256,128] ELU, lr 1e-3 adaptive(KL 0.01), γ 0.99, λ 0.95, clip 0.2, entropy 0.003 |
| Export | play.py で JIT/ONNX、train.py で `params/deploy.yaml`（C++ deploy 用） |

## 3. Go2-W URDF の 16 駆動関節（URDF から機械的に抽出）

`scripts/urdf_audit.py` で抽出（全表は `docs/URDF_AUDIT_go2w_generated.md`）。

| # | joint | type | parent → child | axis | lower | upper | effort [Nm] | vel [rad/s] | 区分 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | FL_hip_joint | revolute | base → FL_hip | 1 0 0 | −1.0472 | 1.0472 | 23.7 | 30.1 | 脚 |
| 1 | FL_thigh_joint | revolute | FL_hip → FL_thigh | 0 1 0 | −1.5708 | 3.4907 | 23.7 | 30.1 | 脚 |
| 2 | FL_calf_joint | revolute | FL_thigh → FL_calf | 0 1 0 | −2.7227 | −0.8378 | 35.55 | 20.07 | 脚 |
| 3–5 | FR_{hip,thigh,calf}_joint | 同上 | | | 同上 | | | | 脚 |
| 6 | RL_hip_joint | revolute | base → RL_hip | 1 0 0 | −1.0472 | 1.0472 | 23.7 | 30.1 | 脚 |
| 7 | RL_thigh_joint | revolute | RL_hip → RL_thigh | 0 1 0 | **−0.5236** | **4.5379** | 23.7 | 30.1 | 脚 |
| 8 | RL_calf_joint | revolute | RL_thigh → RL_calf | 0 1 0 | −2.7227 | −0.8378 | 35.55 | 20.07 | 脚 |
| 9–11 | RR_{hip,thigh,calf}_joint | 同上 | | | 同上 | | | | 脚 |
| 12 | FL_foot_joint | **continuous** | FL_calf → FL_foot | 0 1 0 | – | – | 23.7 | 30.1 | 車輪 |
| 13 | FR_foot_joint | continuous | FR_calf → FR_foot | 0 1 0 | – | – | 23.7 | 30.1 | 車輪 |
| 14 | RL_foot_joint | continuous | RL_calf → RL_foot | 0 1 0 | – | – | 23.7 | 30.1 | 車輪 |
| 15 | RR_foot_joint | continuous | RR_calf → RR_foot | 0 1 0 | – | – | 23.7 | 30.1 | 車輪 |

URDF 上の事実:

- **車輪関節名は `*_foot_joint`、車輪リンク名は `*_foot`**（Go2 では同名の関節が `fixed`）。Go2 の報酬/センサで `.*_foot` を「足先」とみなしている箇所は、Go2-W では「車輪」になる。
- 車輪軸はすべて `+y`（左右とも同じ）。正の角速度で前進方向に転がる（ω ŷ × r ẑ = ωr x̂）。→ Joint Test で要確認。
- 車輪半径 **0.0860 m**、幅 0.0518 m、関節原点から横方向に ±0.048 m オフセット（衝突メッシュ `left/right_wheel.dae` の外形から算出）。車輪 CoM も y=±0.06 にオフセット。
- 車輪の collision は **メッシュ**（dae）。Isaac Lab の URDF インポータ既定は `convex_hull` なので多角形近似の車輪になる → 転がり品質を Joint Test で確認し、必要なら円柱 collision に差し替えたパッチ URDF を使う。
- 前脚と後脚で thigh の可動範囲が異なる。calf の effort/velocity は 35.55 Nm / 20.07 rad/s（Go2 の URDF は 45.43 / 15.70 で、**Go2 と Go2-W で calf の減速比が異なる**可能性）。
- 総質量 19.52 kg（Go2 URDF 16.09 kg）。base 6.921 kg。各 `*_foot_motor`（0.646 kg, fixed）＋車輪 `*_foot`（0.52 kg）。
- `calflower`, `calflower1` は inertial なし・collision ありの fixed リンク（インポート時に calf にマージされる想定）。
- `config/joint_names_go2w_description.yaml` に `"Fl_thigh_joint"` という typo（URDF 本体は `FL_thigh_joint`）。本実装はこの yaml を使わない。
- 名目立位（thigh 0.8, calf −1.5）の車体高さ ≈ **0.408 m**（thigh 0.213 m、calf→車輪中心 0.2264 m、車輪半径 0.086 m から計算）。

Isaac Sim 上の関節順序（PhysX の articulation 順）は URDF の記述順とは限らない。Unitree の C++ deploy は「sim の関節順で先頭 12 = 脚、末尾 4 = 車輪」を暗黙に仮定しているため、**Joint Test で実際の順序を出力して確認し、deploy.yaml には関節名も明示的に書き出す**。

## 4. Go2 環境から再利用できるもの

| 再利用 | 方法 |
|---|---|
| Isaac Lab 標準 MDP 関数（`base_ang_vel`, `projected_gravity`, `joint_pos_rel`, `joint_vel_rel`, `height_scan`, `last_action`, `track_*_exp`, `flat_orientation_l2`, `lin_vel_z_l2`, `is_terminated`, `illegal_contact`, `bad_orientation`, `terrain_levels_vel`, events 群） | `isaaclab.envs.mdp` / `isaaclab_tasks...velocity.mdp` から直接 import |
| Unitree の T-N カーブ + 摩擦付き Delayed PD アクチュエータの考え方 | 汎用 QDD モデルとして再実装（Kt・減速比・最大電流から T-N カーブを生成できる形に一般化） |
| PPO ハイパーパラメータ・ネットワーク構造 | `BasePPORunnerCfg` と同値を初期値に採用 |
| train/play スクリプトの流れ、JIT/ONNX export、`deploy.yaml` 形式 | 同形式で再実装（C++ `deploy/robots/go2w` がそのまま読める形式を維持） |
| `lin_vel_cmd_levels` 型のコマンドカリキュラム | 汎用化して再実装（x/y/yaw 個別、YAML で閾値設定） |

## 5. Go2-W 向けに変更が必要なもの

| 項目 | Go2 | Go2-W で必要な変更 |
|---|---|---|
| Action | `JointPosition(".*")` 12 次元 | `JointPosition(脚12)` + `JointVelocity(車輪4)` の 2 項 = 16 次元 |
| Actuator | 全関節同一 PD | 脚: 位置 PD（hip/thigh と calf で別パラメータ）、車輪: 速度制御（Kp=0, Kd>0） |
| Obs | `joint_pos_rel(".*")` | 車輪は continuous なので **位置を観測しない**（無限に増加）。脚 12 の位置・速度 + 車輪 4 の速度 |
| 足先報酬 | `feet_air_time`, `feet_slide`, `air_time_variance`（body `.*_foot`） | そのまま使うと「車輪を浮かせろ」「車輪を滑らせるな（=転がるな）」という誤った報酬になる。**Phase 1 では使わない**。後段で「脚を使ったときだけ効く」形に再設計（段差 Phase） |
| feet_slide | 接地足の水平速度を罰する | 車輪は転がって水平速度を持つのが正常 → 「転がり速度 ωr と接地点速度の差」= **wheel slip** に置換 |
| joint_position_penalty / stand_still | `joint_pos - default` を全関節 | 車輪は除外（脚 12 のみ） |
| energy / torque 系 | 全関節 | 脚と車輪を分けて重み付け可能にする（車輪は常時回転するため） |
| undesired_contacts | Head, hip, thigh, calf | base/hip/thigh（calf は階段で当たり得るので後段で検討） |
| 初期姿勢 | hip ±0.1, thigh 0.8/1.0, calf −1.5, z=0.4 | thigh 0.8, calf −1.5, 車輪 0, z≈0.45（名目 0.408 + 余裕） |

## 6. 新規作成ファイル

責務分離（Robot / Actuator / RL ロジック / Deploy）を維持しつつ、Isaac Lab の外部拡張パッケージの形に合わせた構成。**unitree_rl_lab に依存しない独立パッケージ**（Go2-W 専用コードは `robots/go2w.py` と `envs/go2w/` に閉じ込める）。

```text
wheel_legged_rl/
├── README.md
├── docs/{IMPLEMENTATION_PLAN.md, URDF_AUDIT_go2w_generated.md, DEV_LOG.md}
├── scripts/
│   ├── urdf_audit.py         # [Pure Python] URDF → 関節/リンク/車輪半径/立位高さ
│   ├── joint_test.py         # [Isaac] inspect: USD 化後の関節順・リミット・質量・CoM・慣性を URDF と照合
│   │                         #         joints: 各関節に正方向指令 → 回転方向・URDF軸との一致・追従を自動判定
│   │                         #         stand/roll: 立位の接地体・車高、車輪正転で前進するか・有効半径
│   ├── train.py              # [Isaac] PPO 学習（W&B / resume / 前Phaseからの初期化 / YAML 上書き）
│   ├── play.py               # [Isaac] 再生 + キーボード速度指令 + ONNX/JIT export
│   ├── evaluate.py           # [Isaac] 定量評価 → JSON/CSV + W&B
│   └── export_policy.py      # [Isaac不要] checkpoint → TorchScript / ONNX + deploy.yaml
├── source/wheel_legged_rl/wheel_legged_rl/
│   ├── robots/
│   │   ├── robot_config.py   # RobotSpec（ロボット非依存のデータ定義）
│   │   ├── go2w.py           # GO2W_SPEC（URDFから確認した値のみ）
│   │   └── builder.py        # RobotSpec → Isaac Lab ArticulationCfg
│   ├── actuators/
│   │   ├── qdd_model.py      # [Pure] QDD 物理パラメータ → T-N カーブ・摩擦
│   │   ├── leg_actuator.py   # 位置PD + T-N 制限 + 遅延 + 摩擦（Isaac ActuatorBase 派生）
│   │   └── wheel_actuator.py # 速度制御 + T-N 制限 + 遅延 + 摩擦
│   ├── commands/velocity_commands.py
│   ├── observations/observations.py   # 車輪速度, 地形知覚インターフェース（height scan）など
│   ├── rewards/rewards.py             # wheel slip, 脚/車輪分離エネルギー, stand still など
│   ├── terrains/terrain_config.py     # Phase ごとの TerrainGeneratorCfg ファクトリ
│   ├── curriculum/curriculum.py       # コマンド範囲・地形レベル
│   ├── randomization/                 # （Step 9 で追加）
│   ├── envs/
│   │   ├── wheel_legged_env.py        # ManagerBasedRLEnv 派生（診断メトリクス → W&B）
│   │   ├── base_env_cfg.py            # ロボット非依存の Env 設定（RobotSpec から組み立て）
│   │   └── go2w/{__init__.py(gym登録), env_cfg.py}
│   ├── configs/go2w/phase1_flat.yaml  # 報酬重み・コマンド範囲・地形・DR を外部設定化
│   ├── agents/rsl_rl_ppo_cfg.py
│   ├── evaluation/{metrics.py, diagnostics.py}
│   ├── deployment/{policy_interface.py, export_deploy_cfg.py}
│   └── utils/config_loader.py         # YAML → configclass 上書き
└── tests/                              # Isaac 不要の単体テスト（pytest）
```

## 7. Observation（最終候補）

**Policy（実機でも取得できるもののみ）** — 全 Phase で次元を固定し、Phase 間で重みを引き継げるようにする。

| 項目 | 次元 | スケール | ノイズ(一様) | 実機での取得元 |
|---|---|---|---|---|
| base_ang_vel | 3 | 0.25 | ±0.2 | IMU ジャイロ |
| projected_gravity | 3 | 1 | ±0.05 | IMU 姿勢 |
| velocity_commands [vx, vy, ωz] | 3 | 1 | – | 上位 ECU |
| joint_pos_rel（脚 12 のみ。車輪は continuous なので除外） | 12 | 1 | ±0.01 | エンコーダ |
| joint_vel_rel（脚 12 + 車輪 4 を 1 項目で。C++ deploy の登録名に合わせる） | 16 | 0.05 | ±1.5 | エンコーダ |
| last_action | 16 | 1 | – | 内部状態 |
| terrain_height_scan | 187 | 1 | ±0.02 (Phase 1) | Perception module（LiDAR/Depth → elevation map） |
| **合計** | **240** | | | |

- `terrain_height_scan`: 車体 yaw 整列の 1.6 m × 1.0 m グリッド、0.1 m 間隔（17×11=187）、値は `base_z − hit_z − 0.5` を [−1, 1] にクリップ。**「Perception module の出力 = ロボット中心の高さマップ」というインターフェース**として定義し、シミュレーションでは RayCaster、実機では LiDAR/Depth の elevation map で同じテンソルを埋める。Phase 1（平地）でも入力に含めておき、次元を変えない。後段でノイズ・欠損（ドロップアウト）・オフセット・遅延をランダム化して完全情報への依存を防ぐ。
- 関節の順序は **シミュレータの関節順（脚のみ／車輪のみを抽出）**。Unitree の C++ deploy が「action i = sim 関節 i」を前提としているため、それと互換にする。実際の順序と関節名は `deploy.yaml`（`actions.*.joint_names`, `robot.sim_joint_names`）に明示的に書き出され、`PolicyInterface` はそれを読むので、ロボットを替えて順序が変わっても推測は不要。`joint_test.py` が「脚=0..11、車輪=12..15」かどうかを検査する。
- 履歴（history_length）は設定値。Phase 1 は 1。

**Critic（非対称、特権情報可）**: Policy と同じ項目（ノイズなし）+ base_lin_vel(3) + 全関節トルク(16) + 車輪接地フラグ(4) = 263 次元。

## 8. Action（最終定義）

```text
action[0:12]  = JointPositionAction (legs)  : q_target = q_default + 0.25 * a        (rad)
action[12:16] = JointVelocityAction (wheels): ω_target = 5.0 * a, clip ±30.1 rad/s   (rad/s)
```

- Isaac Lab の `ActionManager` は項目の定義順に連結するので、`leg_pos`（`JointPositionActionCfg`）→ `wheel_vel`（`JointVelocityActionCfg`）の順で定義すれば上記レイアウトになる（v2.3.0 ソースで確認）。
- 関節順は sim の関節順（§7 参照。deploy.yaml に名前付きで出力）。Action 項目名は Unitree C++ deploy の登録名 `JointPositionAction` / `JointVelocityAction` に合わせる。
- スケール・クリップは設定値。車輪スケール 10 rad/s ≈ 0.86 m/s（r=0.086）。上限 30 rad/s ≈ 2.6 m/s は URDF の velocity limit 30.1 に一致。
- Low-level: 脚 `τ = Kp(q* − q) + Kd(0 − q̇)`、車輪 `τ = Kd_w(ω* − ω)`、いずれもアクチュエータモデルでトルク制限。

## 9. Reward 初期構成（すべて YAML で重み・パラメータ変更可、`null` で無効化）

| 項 | 関数 | Phase 1 重み | 追加予定 Phase |
|---|---|---|---|
| 速度追従 (xy) | `track_lin_vel_xy_exp` (σ=0.5) | +1.5 | 1 |
| Yaw 追従 | `track_ang_vel_z_exp` (σ=0.5) | +0.75 | 1（指令は 0 固定なので「曲がらない」効果） |
| 姿勢（roll/pitch） | `flat_orientation_l2` | −2.5 | 1 |
| 上下動 | `lin_vel_z_l2` | −2.0 | 1 |
| roll/pitch 角速度 | `ang_vel_xy_l2` | −0.05 | 1 |
| 転倒ペナルティ | `is_terminated`（time_out 除く） | −10 | 1 |
| 静止 | `stand_still`（指令≈0 のとき脚偏差 + 車輪速度） | 0（無効） | 1b/2 |
| Action rate | `action_rate_l2` | 0 | 2 |
| エネルギー（脚/車輪別） | `mechanical_power`（|τ·ω|） | 0 | 3 |
| 関節トルク | `joint_torques_l2`（脚） | 0 | 3 |
| 関節リミット | `joint_pos_limits`（脚） | 0 | 2 |
| Wheel slip | `wheel_slip`（接地車輪の |ωr − v_long|²） | 0 | 3 |
| 車体高さ | `base_height_l2`（地形相対、height scan 利用） | 0 | 3 |
| 衝突 | `undesired_contacts`（base/hip/thigh） | 0 | 2 |
| 地形踏破 | `terrain_levels_vel` によるカリキュラム + 進行距離報酬 | – | 4 |

ソースコードには重みの既定値を書かず、`configs/go2w/phase*.yaml` がすべての重みを持つ。未記載の項目は「無効」扱い。CLI の Hydra 上書き（`env.rewards.action_rate.weight=-0.01`）も利用可能。

## 10. Actuator 構成

```text
RobotSpec.actuators = {
  "hip_thigh": LegActuatorSpec(joints=[.*_hip_joint, .*_thigh_joint], Kp=25, Kd=0.5, QDD 曲線, delay 0..N)
  "calf":      LegActuatorSpec(joints=[.*_calf_joint],               Kp=25, Kd=0.5, QDD 曲線, delay 0..N)
  "wheel":     WheelActuatorSpec(joints=[.*_foot_joint],             Kd=0.5, QDD 曲線, delay 0..N)
}
```

- 実装は Isaac Lab の explicit actuator（`DelayedPDActuator` 派生）。**トルク計算を Python 側で行う**ので、実機 ECU の制御則と 1:1 で置き換えられる。PhysX 側のドライブゲインは 0。
- **QDD モデル**（`actuators/qdd_model.py`、Isaac 非依存）: `QDDParams(kt, gear_ratio, i_max, motor_no_load_speed, motor_knee_speed, rotor_inertia, gear_efficiency, coulomb_friction, viscous_friction)` → 出力軸の `Y1/Y2/X1/X2`（T-N カーブ）、armature（反射慣性 = J_rotor·N²）を計算。Unitree の T-N 表記で直接与えることもできる。
- Go2-W の値:
  - hip/thigh: Unitree `Go2HV` 曲線（Y1 20.2, Y2 23.4, X1 13.5, X2 30）＝ unitree_rl_lab の値。URDF effort 23.7/vel 30.1 と整合。
  - calf: **推定** — Go2HV 曲線 × 減速比 1.5（トルク×1.5、速度÷1.5 → Y1 30.3, Y2 35.1, X1 9.0, X2 20.0）。URDF の calf effort 35.55 / vel 20.07 と整合するが Unitree 公式値ではないため要確認。
  - 車輪: **推定** — URDF の effort 23.7 / vel 30.1 から Go2HV と同一曲線を仮定。Kd=0.5 は unitree_rl_lab の `UNITREE_GO2W_CFG` の値。
- 実機との対応: `deploy.yaml` に Kp/Kd を関節ごとに出力（C++ deploy と同じ形式）。

## 11. Terrain 構成（`terrains/terrain_config.py`、Phase ごとに比率を YAML 指定）

| 地形 | Isaac Lab 実装 | ランダム化（difficulty に比例） |
|---|---|---|
| 平地 | `MeshPlaneTerrainCfg` | 摩擦（ロボット側 material 乱択, combine=multiply） |
| 小段差 | `MeshRandomGridTerrainCfg`（grid_height_range） / `MeshBoxTerrainCfg` | 高さ |
| 階段 上り/下り | `MeshPyramidStairsTerrainCfg` / `MeshInvertedPyramidStairsTerrainCfg` | step_height, step_width（深さ）。段数はサブ地形サイズと platform_width で決まる |
| スロープ 上り/下り | `HfPyramidSlopedTerrainCfg` / `HfInvertedPyramidSlopedTerrainCfg` | 傾斜、長さ（サブ地形サイズ・platform_width） |
| 凹凸 | `HfRandomUniformTerrainCfg`, `HfWaveTerrainCfg` | 高さ・間隔（noise_step, downsampled_scale）・密度 |
| 砂利相当 | `HfRandomUniformTerrainCfg`（小振幅・高周波）＋ 低摩擦 | 石サイズ（horizontal_scale/downsampled_scale）、高さ、摩擦 |
| 水たまり/低摩擦 | 地形ごと摩擦の低い sub-terrain 列（env の terrain type に応じて摩擦を再設定するイベント）＋ 車輪粘性抵抗項 | 摩擦係数、「水深相当」→ 車輪への速度比例抵抗トルク |
| 障害物 | `HfDiscreteObstaclesTerrainCfg`, `MeshRepeatedBoxesTerrainCfg` | 高さ・幅・長さ・位置・数 |
| 複合（Phase 7） | 独自 `SubTerrainBaseCfg`（1 タイル内に Flat→Slope→Gravel→Stairs→Flat を直列配置する mesh 関数） | 各区間の difficulty |

Isaac Lab の TerrainGenerator は「行 = 難易度、列 = 地形種類」。`terrain_levels_vel` で行（難易度）を上げ下げする。Policy には Terrain ID を与えない（Critic には与えてもよい）。

## 12. Curriculum 構成

**2 層構成**

1. **Phase（外側、手動 / スクリプト）**: タスク ID ごとに YAML を切替え、前 Phase の checkpoint から初期化（`train.py --init_from <ckpt>`）。Obs/Action 次元は全 Phase 共通なのでそのまま fine-tuning 可能。
2. **Phase 内（自動）**: コマンド範囲カリキュラム（追従報酬が閾値超えで範囲拡大）、地形レベルカリキュラム（`terrain_levels_vel`）。

| Phase | タスク ID | 地形 | コマンド | 主な追加報酬 | 完了条件（evaluate.py で判定） |
|---|---|---|---|---|---|
| 1 | `WL-Go2W-Phase1-Flat-v0` | 平地 | vx∈[−1,1], vy=0, ωz=0, 静止 20% | 追従・姿勢・転倒 | vx RMSE < 0.15 m/s、転倒率 < 1%、静止時残留速度 < 0.05 m/s |
| 2 | `...-Phase2-Flat3Axis-v0` | 平地 | +vy∈[−0.5,0.5], ωz∈[−1.5,1.5] | action_rate, joint_limits, collision, stand_still | 3 軸 RMSE 閾値 |
| 3 | `...-Phase3-FlatCoord-v0` | 平地 + 摩擦乱択 | 同上 + 高速域 | energy(脚/車輪), wheel slip, base height | 車輪走行時の姿勢 RMS < 閾値 |
| 4 | `...-Phase4-Steps-v0` | 平地 + 小段差(0→0.15 m) | 同上 | terrain progress | 段差踏破率 > 80% |
| 5 | `...-Phase5-StairsSlope-v0` | + 階段・スロープ | | 脚報酬（接地時間は段差/階段のみ） | 階段成功率 |
| 6 | `...-Phase6-Rough-v0` | + 凹凸・砂利・低摩擦 | | | |
| 7 | `...-Phase7-Mixed-v0` | + 複合地形 | | | 未学習地形での汎化 |

**忘却対策**: 各 Phase の TerrainGenerator に前 Phase の地形を一定割合（例: 平地 ≥ 20%、以前の地形合計 ≥ 30%）残す（sub-terrain の proportion で指定）。コマンドも過去 Phase の分布を含める。評価は全 Phase の地形で毎回実施し、過去性能の低下を W&B で監視する。

## 13. Training / Evaluation フロー

```text
urdf_audit.py → joint_test.py (inspect, joints, stand, roll)   # モデル検証（RL の前に必須）
      ↓
train.py --task WL-Go2W-Phase1-Flat-v0 --headless --logger wandb
      ↓ (checkpoints: logs/rsl_rl/go2w_wheel_leg/<date>_phase1/model_*.pt, best は evaluate で選定)
evaluate.py --task ... --checkpoint ... → metrics.json + W&B
      ↓
play.py --task ... --keyboard        # 目視確認 + ONNX/JIT export
      ↓
train.py --task WL-Go2W-Phase2-... --init_from <phase1 ckpt>
```

- Resume: `--resume --load_run <run> --checkpoint model_XXXX.pt`（同一 Phase の中断再開、optimizer も復元）。
- 前 Phase からの初期化: `--init_from <path>`（重みのみ、iteration は 0 から）。
- Best model: `evaluate.py` が checkpoint 群を評価し `best.pt` へコピー（学習中の報酬最大ではなく評価指標で選ぶ）。

評価指標（`evaluation/metrics.py`、Isaac 非依存で単体テスト済み）: 速度 RMSE、yaw RMSE、成功率、転倒率、停止時残留速度、roll/pitch RMS、移動距離あたりエネルギー（CoT）、wheel slip ratio、未学習地形の成功率。

## 14. W&B 連携

- rsl_rl 3.0.1 の `logger="wandb"`（`--logger wandb --log_project_name <proj>`）。PPO 指標（value/surrogate loss, entropy, KL, lr, action std）と `extras["log"]` の全キーが自動記録される。
- 自動で入るもの: `Episode_Reward/<各報酬項>`、`Episode_Termination/<各終了条件>`、`Curriculum/<各項>`、`Metrics/base_velocity/error_vel_xy, error_vel_yaw`。
- 追加（`envs/wheel_legged_env.py` の診断機能）: `Diagnostics/fall_rate`, `wheel_slip_ratio`, `energy_per_meter`, `mech_power_leg/wheel`, `action_rate`, `terrain_level_mean`, `wheel_contact_ratio`、さらに地形種別ごとの `Diagnostics/<terrain>/vel_error`（Phase 4 以降で有効）。
- evaluate.py は同じ W&B project に `eval/` プレフィックスで書き込む。

## 15. 自作ロボットへの置換のための抽象化

```text
RobotSpec (robots/robot_config.py)         ← ロボット固有（Go2-W / 自作機）
├─ asset: urdf_path | usd_path, collider 設定
├─ leg_joint_names[12], wheel_joint_names[4], wheel_body_names[4], base_body_name
├─ joint_limits は URDF から（上書き可）
├─ default_joint_pos, init_base_height
├─ wheel_radius
├─ actuators: {group: LegActuatorSpec | WheelActuatorSpec(QDDParams or T-N, Kp, Kd, delay)}
├─ undesired_contact_bodies, termination_contact_bodies
└─ sdk_joint_order (実機 ECU の関節並び)

builder.py:  RobotSpec → ArticulationCfg（Isaac 固有処理はここだけ）
base_env_cfg.py: RobotSpec を受けて Scene/Obs/Action/Reward/Termination を組み立て（ロボット非依存）
configs/<robot>/phase*.yaml: 報酬・コマンド・地形・DR
deployment/policy_interface.py: Isaac 非依存。deploy.yaml + ONNX で
    obs = build_obs(imu, joint_state, command, height_map, last_action) → action → (q*, ω*)
    を行い、MuJoCo（Sim2Sim）と実機 ECU で同じコードを使う
```

自作機への移行手順: ① `urdf_audit.py` で URDF 検証 → ② `robots/custom.py` に `RobotSpec` を 1 つ書く（関節名、車輪半径、QDD パラメータ）→ ③ `envs/custom/` で gym 登録（数行）→ ④ `configs/custom/phase*.yaml` をコピーして調整 → ⑤ `joint_test.py` → 学習。RL ロジック（obs/reward/terrain/curriculum）は変更不要。

---

## 本セッションでの実装範囲と検証方法

- **実装**: 上記アーキテクチャの骨格 + Phase 1（平地・停止/前進/後退）まで完全に。Phase 2 以降は YAML とタスク ID の追加で進められる形にしておくが、Phase 1 の実機動作確認前には有効化しない（方針1）。
- **本セッションの制約**: 作業環境に GPU / Isaac Sim がないため、Isaac Sim 上での実行（USD 変換・Joint Test・学習）はここではできない。代わりに
  1. Isaac Lab v2.3.0 のソースと突き合わせた API 静的確認（使用する全クラス・関数・引数名）
  2. Isaac 非依存部分（URDF 監査、QDD モデル、YAML ローダ、報酬の数式、評価指標、Policy Interface、deploy.yaml 互換性）の pytest
  3. 全ファイルの構文チェック
  を実施する。Isaac Sim 上での確認手順は README の「初回動作確認チェックリスト」にまとめる。
