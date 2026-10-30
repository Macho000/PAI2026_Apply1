# wheel_legged_rl — 四足四輪ロボットの RL 環境（Go2-W 版）

Isaac Lab 上で、脚 12 自由度を位置制御し、車輪 4 自由度を速度制御する四足四輪ロボットの歩行・走行ポリシーを学習する環境です。最初の対象は Unitree Go2-W です。ロボット固有の情報は `RobotSpec` 1 つにまとめており、自作機（Quadra）へ移すときに RL ロジックは変更しません。

- 設計の根拠: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
- 実装時の判断、計画との差分、検証結果: [docs/DEV_LOG.md](docs/DEV_LOG.md)
- Go2-W URDF の監査結果: [docs/URDF_AUDIT_go2w_generated.md](docs/URDF_AUDIT_go2w_generated.md)

現在の実装範囲は Phase 1（平地での停止・前進・後退）、Phase 2（平地での 3 軸指令: vx・vy・yaw rate）、Phase 3（高速域 ±2 m/s、エネルギー報酬、Sim-to-Real のドメインランダム化）、Phase 4（小段差 0〜0.15 m と地形カリキュラム）です。Phase 5 以降も、YAML とタスク ID を追加すれば進められる構成にしてあります。

## 対象バージョン

| 要素 | バージョン |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | v2.3.0（API はこのタグのソースで静的に確認済み） |
| rsl-rl-lib | 3.0.1 |
| Python | 3.11（Isaac Sim 同梱）。Isaac に依存しない部分は 3.10 以上で動作 |

## 構成

```text
wheel_legged_rl/
├── scripts/
│   ├── fetch_go2w_description.sh  Go2-W URDF とメッシュを commit 固定で取得（third_party/、git 管理外）
│   ├── urdf_audit.py         [Isaac 不要] URDF の関節・質量・車輪径・立位高さの監査と RobotSpec との照合
│   ├── joint_test.py         [Isaac] 関節順・リミット・慣性・回転方向・立位・転がりの自動検査
│   ├── train.py              [Isaac] PPO 学習（W&B / resume / --init_from / --env_config）
│   ├── play.py               [Isaac] 再生、キーボード指令、TorchScript/ONNX 出力
│   ├── evaluate.py           [Isaac] シナリオ評価 → metrics.json/csv、W&B、best.pt の選定
│   ├── export_policy.py      [Isaac 不要] checkpoint → TorchScript/ONNX（deploy.yaml と次元を照合）
│   └── check_isaaclab_api.py [Isaac 不要] Isaac Lab / rsl_rl ソースとの API 静的照合
├── source/wheel_legged_rl/wheel_legged_rl/
│   ├── robots/        RobotSpec（robot_config.py）、GO2W_SPEC（go2w.py）、Isaac 変換（builder.py）
│   ├── actuators/     QDD モデル（Isaac 非依存）、脚・車輪の explicit actuator
│   ├── envs/          ロボット非依存の Env 設定、診断付き Env、go2w タスク登録
│   ├── configs/go2w/  phase1_flat.yaml（報酬重み・コマンド・地形・DR・評価基準）
│   ├── observations/ rewards/ commands/ curriculum/ terrains/
│   ├── evaluation/    metrics.py（評価指標）、diagnostics.py（学習中の診断）
│   ├── deployment/    deploy.yaml 出力、PolicyInterface（Sim2Sim / 実機共通）、checkpoint 変換
│   └── utils/         URDF 解析、YAML ローダ
└── tests/             Isaac 不要の pytest
```

## セットアップ（Docker、推奨）

`docker/` に Isaac Sim 5.1.0 + Isaac Lab v2.3.0 のコンテナを用意しています。NGC 公式イメージ `nvcr.io/nvidia/isaac-lab:2.3.0` に rsl-rl-lib 3.0.1、onnx、onnxruntime、wandb 0.22.3 以上を追加した構成で、`ur10-robotiq-grasp` の Docker 構成に合わせています。quadra リポジトリ全体を `/quadra` にマウントするので、ホスト側で編集した内容はすぐにコンテナへ反映されます。NVIDIA Container Toolkit が必要です。

```bash
./scripts/fetch_go2w_description.sh          # ホスト側で Go2-W の URDF を取得
cd docker
./setup_host.sh                              # 初回のみ: ~/docker/isaac-sim-5.1 にキャッシュ用ディレクトリを作成、xhost を設定
cp .env.example .env                         # W&B を使う場合は WANDB_API_KEY を記入
docker compose build                         # 初回はベースイメージ 8.5 GB を取得

./wl.sh joint_test.py --robot go2w --headless              # scripts/ 内のスクリプトをコンテナ内で実行
./wl.sh train.py --task WL-Go2W-Phase1-Flat-v0 --headless
./wl.sh -m pytest                                          # Isaac Lab 同梱の Python で単体テストを実行
./wl.sh bash                                               # コンテナのシェルに入る
```

コンテナは root で動くため、`logs/` などコンテナが作ったファイルは root 所有になります。必要に応じて `sudo chown -R $USER logs` で所有者を戻してください。Isaac Sim のキャッシュは Isaac Sim 6.0 用の `~/docker/isaac-sim` とは分けて、`~/docker/isaac-sim-5.1` に保存します。

## セットアップ（既存の Isaac Lab に入れる場合）

```bash
# 1. Go2-W の URDF とメッシュを取得（unitree_ros @ ccfc6fd、約 30 MB）
./scripts/fetch_go2w_description.sh

# 2. Isaac Lab v2.3.0 の環境にパッケージを入れる
cd "$ISAACLAB_ROOT"
./isaaclab.sh -p -m pip install -e /path/to/quadra/wheel_legged_rl/source/wheel_legged_rl
```

URDF を別の場所に置く場合は、`WL_GO2W_URDF=/path/to/go2w_description.urdf` を設定します。インポート用の URDF（メッシュ絶対パス化と車輪の円柱化を施したもの）と USD は `~/.cache/wheel_legged_rl/` に生成されます。生成先は `WL_CACHE_DIR` で変更できます。

## Isaac 不要の検証

```bash
cd wheel_legged_rl
python -m pip install pytest torch numpy pyyaml gymnasium onnx onnxruntime   # 任意の venv
python -m pytest                                                  # 単体テスト
python scripts/urdf_audit.py --robot go2w --out docs/URDF_AUDIT_go2w_generated.md
python scripts/check_isaaclab_api.py --isaaclab /path/to/IsaacLab-v2.3.0 --rsl_rl /path/to/rsl_rl-3.0.1
```

## 初回動作確認チェックリスト（Isaac Sim 上）

上から順に実行し、結果を [DEV_LOG](docs/DEV_LOG.md) に記録してください。Go2-W では、2026-09-24 に Docker 上で 1〜5 と 7〜9 を実施済みです（joint_test は 31/31 PASS）。6 の本学習は未実施です。Docker の場合は `docker/wl.sh <script> ...` で、既存の Isaac Lab の場合は `$ISAACLAB_ROOT` で `./isaaclab.sh -p <repo>/wheel_legged_rl/scripts/...` で実行します。

1. **モデル検査**: `joint_test.py --robot go2w --headless --mode inspect --out inspect.json`
   - `legs are sim joints 0..11` と `wheels are sim joints 12..15` が PASS であること。FAIL の場合、Unitree の C++ `deploy/robots/go2w`（先頭 12 = q、末尾 4 = dq と決め打ち）はそのままでは使えません。`PolicyInterface` は deploy.yaml の関節名を読むため、影響を受けません。
   - 質量・CoM・慣性・リミットが URDF と一致すること。比較用の参照は、インポータが実際に残したボディに合わせて組み立てます（Isaac Sim 5.1 は質量を持つ fixed リンクをマージしません）。PhysX のドライブゲインが 0 であること。
2. **関節方向**: `--mode joints`。各脚関節に +0.2 rad、各車輪に +5 rad/s を与え、回転方向が URDF 軸（右手系）と一致することを確認します。
3. **立位**: `--mode stand`（FixStand 相当のゲイン、車輪は位置保持）。車体高さが、実測の関節角から計算した URDF の FK と 1 cm 以内で一致すること。姿勢が水平で、接地しているのが車輪 4 つだけであること。
4. **転がり**: `--mode roll`。車輪の正転で前進し、有効半径が 0.086 m ± 10% に収まること。`--mesh_wheels` を付けると、元のメッシュ（convex hull）のタイヤと比較できます。
5. **学習の煙テスト**: `train.py --task WL-Go2W-Phase1-Flat-v0 --headless --num_envs 64 --max_iterations 5`
   - ログに `disabled zero-weight rewards: [...]` と `sim joint order: [...]` が出ること。
   - `logs/rsl_rl/go2w_wheel_leg/<run>/params/deploy.yaml` の `policy.obs_dim` が 240、`action_dim` が 16 であること。
6. **本学習**: `train.py --task WL-Go2W-Phase1-Flat-v0 --headless --logger wandb --log_project_name wheel_legged_rl`
   - RTX 3060（12 GB）で VRAM が不足する場合は `--num_envs 2048` にします。
   - W&B で `Curriculum/lin_vel_x_levels`（コマンド範囲の幅）が 1.0 から 2.0 へ広がることを確認します。
7. **評価と best 選定**: `evaluate.py --load_run <run> --all_checkpoints --every 5 --select_best --wandb --headless`。Phase 1 の完了条件（vx RMSE < 0.15 m/s、転倒率 < 1%、静止時の残留速度 < 0.05 m/s）を満たすと、終了コードが 0 になります。
8. **目視と出力**: `play.py --load_run <run> --checkpoint <run>/best.pt --keyboard`（↑↓ で vx）。`<run>/exported/policy.onnx` が出力されます。
9. **Isaac 外での推論**: `python scripts/export_policy.py --checkpoint <run>/best.pt`。その後、`PolicyInterface.from_run_dir(<run>)` で 1 ステップ推論できることを確認します。

## 学習フロー

```text
urdf_audit.py → joint_test.py → train.py (Phase 1) → evaluate.py --select_best → play.py → 実機 / Sim2Sim
                                      └→ train.py --task WL-Go2W-Phase2-Flat3Axis-v0 --checkpoint <phase1 run>/best.pt
                                           └→ train.py --task WL-Go2W-Phase3-FlatCoord-v0（Phase 2 の best.pt を自動で使用）
                                                └→ train.py --task WL-Go2W-Phase4-Steps-v0（Phase 3 の best.pt を自動で使用）
```

Phase 2（Phase 1 の checkpoint から学習）:

```bash
./docker/wl.sh train.py --task WL-Go2W-Phase2-Flat3Axis-v0 --headless --logger wandb --log_project_name wheel_legged_rl \
    --checkpoint logs/rsl_rl/go2w_wheel_leg/<phase1 run>/model_3999.pt
./docker/wl.sh evaluate.py --task WL-Go2W-Phase2-Flat3Axis-Play-v0 --load_run <phase2 run> --all_checkpoints --every 5 --select_best --headless
./docker/wl.sh play.py --task WL-Go2W-Phase2-Flat3Axis-Play-v0 --load_run <phase2 run> --checkpoint best.pt --keyboard
```

Phase 3（Phase 2 の best.pt から学習）:

```bash
# 初期値は最新の phase2_flat3axis run の best.pt（先に Phase 2 で evaluate.py --select_best を実行しておく）
./docker/wl.sh train.py --task WL-Go2W-Phase3-FlatCoord-v0 --headless --logger wandb --log_project_name wheel_legged_rl
./docker/wl.sh evaluate.py --task WL-Go2W-Phase3-FlatCoord-Play-v0 --load_run <phase3 run> --all_checkpoints --every 5 --select_best --headless
# Phase 2 と同じモデル条件（ランダム化なし・遅延 0）での評価。best.pt は変えない
./docker/wl.sh evaluate.py --task WL-Go2W-Phase3-FlatCoord-Play-v0 --load_run <phase3 run> --all_checkpoints --every 5 --headless \
    --env_config source/wheel_legged_rl/wheel_legged_rl/configs/go2w/phase3_flatcoord_nominal_eval.yaml \
    --out logs/rsl_rl/go2w_wheel_leg/<phase3 run>/eval/nominal
```

Phase 3 のドメインランダム化（`configs/go2w/phase3_flatcoord.yaml` の `events` / `actuators`）:

| 項目 | 範囲 | 再サンプル |
|---|---|---|
| 指令遅延（全アクチュエータ共通） | 0〜20 ms（physics 0〜4 step） | エピソードごと |
| モータ強度（T-N 曲線のトルク軸） | ×0.8〜1.2（モータごと） | エピソードごと |
| Kp / Kd | ×0.8〜1.2（モータごと） | エピソードごと |
| 関節摩擦（絶対値、推定範囲） | クーロン 0〜0.2 N·m、粘性 0〜0.02 N·m·s/rad（モータごと） | エピソードごと |
| base 質量 | −1〜+3 kg（慣性も質量比で変更） | 起動時 |
| 脚・車輪リンクの質量 | ×0.9〜1.1（慣性も変更） | 起動時 |
| base の慣性 | ×0.8〜1.2 | 起動時 |
| base の重心 | x ±3 cm、y/z ±2 cm | 起動時 |
| 接触摩擦 | 0.3〜1.2（collision shape ごと。4 輪で異なる） | 起動時 |

Phase 1 / 2 の YAML では、摩擦と質量以外のモデル誤差の項を `null` で無効にしています（挙動は従来のまま）。

Phase 3 の critic には、上の値（正規化済み、90 次元）を特権観測として追加しています。actor の観測は変わりません。

- **Phase 2 からの引き継ぎ**: actor はそのまま読み込みます。critic の第 1 層は、Phase 2 の重みを先頭の 263 列にコピーし、追加した 90 列を 0 にします（開始時点では `V_phase3(obs, 特権) = V_phase2(obs)`）。optimizer は読み込みません。
- **critic の warm-up**: 学習を始めると、まず actor（MLP と action std）を `requires_grad=False` で凍結し、critic だけを学習します。この間は学習率を 3e-4 に固定し（adaptive LR を止める）、epoch 数を 2 にします。解除の条件は `Go2WPhase3PPORunnerCfg.critic_warmup` で設定します。critic だけの更新を最低 2 回、最大 20 回行い、その間に次のどちらかを満たしたら解除します。
  - 新しい rollout に対する更新前の EV が頭打ちになり（上昇 < 0.01 が 2 回続く）、EV ≥ 0.5、かつ |平均誤差| / std(R) ≤ 0.1
  - 汎化ギャップ（前回の更新後 EV − 今回の更新前 EV）が 0.1 を超えた（過学習の兆候）
- **ログ**: `Critic/{ev_pre, ev_post, ev_gap, value_mse_pre, value_bias_pre, rel_bias_pre, value_mean, value_std, return_mean, return_std}`（warm-up 後も毎 iteration）、`Warmup/{active, updates, unfreeze_reason}`（1 = converged、2 = ev_gap、3 = max_iterations）。
- warm-up は `init_from` で始めた run だけで行います（`--resume` やスクラッチでは行いません）。無効にするときは `agent.critic_warmup=null`、固定回数にするときは `min_iterations` と `max_iterations` を同じ値にします。

Phase 4（小段差、Phase 3 の best.pt から学習）:

```bash
./docker/wl.sh train.py --task WL-Go2W-Phase4-Steps-v0 --headless --logger wandb --log_project_name wheel_legged_rl
./docker/wl.sh evaluate.py --task WL-Go2W-Phase4-Steps-Play-v0 --load_run <phase4 run> --all_checkpoints --every 5 --select_best --headless
```

- **地形**（`configs/go2w/phase4_steps.yaml` の `terrain`）: 10 行（難易度）× 20 列。平地 20%（忘却対策）、ランダムグリッド 20%（0.75 m 角のセルが ±0〜0.075 m、隣のセルとの段差は最大 0.15 m）、上りの縁石 30%、下りの縁石 30% です。縁石は Isaac Lab のピラミッド階段で、段の奥行きを 1 m にしたものです。タイル中央の 2 m 四方から、1 m ごとに 3 段あります。段の高さは行に応じて 0〜0.15 m です。学習開始時のロボットは 0〜3 行目に置きます。
- **地形カリキュラム**（`curriculum.terrain_levels_tracking`）: エピソードが終わったロボットのうち、転倒せずにタイルの外（原点から 4 m 以上）まで出たものを 1 行上げます。転倒したもの、または xy 速度の追従報酬がエピソード平均で最大値の 50% 未満だったもの（段差の前で止まった、など）を 1 行下げます。Isaac Lab の `terrain_levels_vel` は移動距離が短いと下げますが、yaw rate 指令で円弧を描いたロボットは原点の近くで終わるため、代わりに追従報酬を使っています。
- **指令・報酬・ドメインランダム化**: Phase 3 と同じです。vx は最初から ±2 m/s です。
- **ログ**: `Diagnostics/<地形>/vel_error`、`Diagnostics/<地形>/terrain_level`（地形の種類ごとの速度誤差と平均の行）、`Curriculum/terrain_levels`。
- **評価**: `evaluation.terrain` が評価用の地形（1 行、段の高さは固定）に置き換わります。各シナリオでは、指定したタイルの中央に全ロボットを置きます。Phase 1〜3 のシナリオは平地（`flat` グループ）で、段差のシナリオ（`steps` グループ）は 1 m/s で 3.5 m 先まで進めたら成功です。対象は上り縁石 0.10 / 0.15 m（0.15 m は後退も）、下り縁石 0.10 / 0.15 m、ランダムグリッド ±0.075 m です。完了条件は、平地で Phase 3 と同じ基準を満たすことと、段差の踏破率（`steps/success_rate`）が 80% 以上であることです。

- 中断した学習の再開（optimizer と iteration も復元）: `train.py --resume --load_run <run> --checkpoint model_1500.pt`
- 前 Phase からの初期化（重みのみ、iteration は 0 から）: `train.py --init_from <path>`。`<path>` は checkpoint ファイル、または run ディレクトリ（その `best.pt`）です。`--resume` なしで `--checkpoint <ファイル>` を渡しても同じです。どちらも指定しないと、runner 設定の `init_from`（Phase 3 は `phase2_flat3axis`: 最新の該当 run の `best.pt`）を使います。スクラッチから学習するときは `agent.init_from=null` を付けます。ホストのパス（`/home/<user>/quadra/wheel_legged_rl/...`）もコンテナ内で解決されます。初期化に使った checkpoint は `params/agent.yaml` と W&B の config（`runner_cfg.init_checkpoint`、`runner_cfg.init_mode` = `init_from` / `resume` / `scratch`）に記録されます。
- 設定の差し替え: `--env_config my_phase.yaml`（`_base_: phase1_flat.yaml` で差分だけ書けます）
- Hydra による上書き（引数の最後に付けます）: `env.rewards.action_rate.weight=-0.01 agent.max_iterations=500`

## 設定（phase YAML）

報酬の重みは、すべて `configs/<robot>/phase*.yaml` に書きます。ソースコードには重みの既定値を置いていません。YAML に書かれていない報酬項は重み 0 になり、Env の生成時に削除されます。そのため Hydra で重みを指定すれば、その項を再度有効にできます。`rewards` / `terrain` / `actuators` / `evaluation` 以外のキーは、Env 設定の属性をそのまま上書きします（例: `commands.base_velocity.rel_standing_envs`）。存在しないキーを書くとエラーになります。

## Observation / Action（全 Phase 共通）

| 区分 | 内容 | 次元 |
|---|---|---|
| policy | base_ang_vel ×0.25, projected_gravity, velocity_commands, joint_pos_rel（脚）, joint_vel_rel（全 16）×0.05, last_action, terrain_height_scan（17×11） | 240 |
| critic | policy と同じ項目（ノイズなし）+ base_lin_vel + joint_effort + wheel_contact | 263 |
| action | `JointPositionAction`（脚 12、q* = q0 + 0.25a）+ `JointVelocityAction`（車輪 4、ω* = clip(10a, ±30.1)） | 16 |

## deploy.yaml と実機

`train.py` は学習を始める前に `params/deploy.yaml` を書き出します。形式は `unitree_rl_lab/deploy` の C++ コントローラが読むもの（`joint_ids_map`、SDK 順の Kp/Kd、actions、observations）の上位互換です。ただし `terrain_height_scan` は Unitree の C++ に登録されていない観測項目なので、`policy.cpp_unsupported` に列挙しています。C++ で動かす場合は、同じ名前の `REGISTER_OBSERVATION` を追加してください。Python の `deployment/policy_interface.py` はこの項目にも対応しており、Sim2Sim と実機 ECU の両方で同じコードを使えます。

## 自作ロボットへの移植

1. `urdf_audit.py --urdf my.urdf --wheel-pattern ".*wheel.*"` で URDF を確認します。
2. `robots/<name>.py` に `RobotSpec` を 1 つ書き、`scripts/_paths.py` の `ROBOTS` に登録します。
3. `envs/<name>/` で `configure_for_robot(cfg, SPEC)` と `apply_phase_config(cfg, yaml)` を呼ぶ設定クラスを作り、gym に登録します（数行）。
4. `configs/go2w/phase1_flat.yaml` をコピーして調整します。
5. `joint_test.py --robot <name>` を通してから学習します。

## ライセンス

Go2-W の URDF とメッシュ（`third_party/unitree_ros`）は Unitree Robotics の BSD-3-Clause ライセンスで、このリポジトリには含めていません。`scripts/cli_args.py` は Isaac Lab（BSD-3-Clause）を基にしています。
