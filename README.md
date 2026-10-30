# PAI2026_Apply1 — 車輪脚ロボット Go2-W の速度指令追従と段差走破の強化学習

Physical AI 応用 1 講座（2026 Summer）最終課題のコード一式です。

Unitree Go2-W は、四脚の先端に車輪をもつロボットです。この課題では、強化学習（PPO）で 1 つの方策を学習し、次の 2 つを両立させました。

- 速度指令 [vx, vy, ωz] への追従
- 0〜0.15 m の段差の走破

シミュレータは NVIDIA Isaac Sim / Isaac Lab です。

## タスクの概要

| 項目 | 内容 |
|---|---|
| ロボット | Unitree Go2-W（19.5 kg、車輪半径 0.086 m） |
| 行動（16 次元） | 脚 12 関節は位置制御（q* = q0 + 0.25a、Kp 25、Kd 0.5）、車輪 4 関節は速度制御（ω* = clip(10a, ±30.1 rad/s)） |
| 観測（240 次元） | IMU 角速度・重力方向、速度指令、関節角度・速度、前回の行動、周囲の高さマップ 17 × 11 |
| 制御周期 | 方策 50 Hz（物理 5 ms × 4） |
| 学習 | PPO（rsl-rl）、4096 台を並列にシミュレーション、MLP 512-256-128 |

学習は 4 段階（Phase）に分けて行い、各 Phase は前の Phase の最良モデルから始めます。

| Phase | 地形 | 速度指令 | 追加した要素 |
|---|---|---|---|
| 1 | 平地 | vx ±1.0 m/s | 静止 → 走行の 2 段階カリキュラム |
| 2 | 平地 | + vy ±0.5 m/s、ωz ±1.5 rad/s | 行動変化率・関節リミット・接触の罰則 |
| 3 | 平地 | vx ±2.0 m/s | Sim-to-Real のドメインランダム化（摩擦、質量、重心、モータ強度、PD ゲイン、指令遅延 0〜20 ms）、エネルギー罰則、critic への特権情報 |
| 4 | 平地 + 段差 0〜0.15 m | Phase 3 と同じ | 地形カリキュラム（難易度 10 段階 × 20 列。平地 20 %、ランダムグリッド 20 %、上り縁石 30 %、下り縁石 30 %） |

## 結果（シミュレーション）

評価はいずれも、物理のランダム化を有効にし、各シナリオ 64 台で行いました。詳細は [wheel_legged_rl/docs/DEV_LOG.md](wheel_legged_rl/docs/DEV_LOG.md) にあります。

| 指標 | 基準 | Phase 3（平地） | Phase 4（平地） | Phase 4（段差） |
|---|---|---|---|---|
| 転倒率 | < 1 % | 0 % | 0 % | 0 % |
| xy 速度 RMSE | < 0.15 m/s | 0.080 m/s | 0.073 m/s | （0.117 m/s） |
| yaw rate RMSE | < 0.2 rad/s | 0.067 rad/s | 0.067 rad/s | （0.122 rad/s） |
| 静止時の残留速度 | < 0.05 m/s | 0.006 m/s | 0.009 m/s | – |
| 姿勢 RMS（roll・pitch） | < 0.05 rad | 0.033 rad | 0.032 rad | （0.076 rad） |
| 段差踏破率 | ≥ 80 % | – | – | 100 %（学習前 43.8 %） |

括弧内の値は基準を適用しない参考値です。段差の評価は、上り・下りの縁石 0.10 / 0.15 m、上り 0.15 m の後退、ランダムグリッド ±0.075 m の 6 シナリオです。1 m/s の指令で 3.5 m 進めたら成功とします。

残っている課題と制約は次のとおりです。

- 2.0 m/s の直進で、指令より遅くなる傾向が残っています（前進 −0.107 m/s、後退 −0.146 m/s）。
- 段差の評価は、学習で使った最大の高さ 0.15 m と奥行 1 m の縁石に限られます。
- 実機ではまだ検証していません。

## リポジトリ構成

```text
PAI2026_Apply1/
└── wheel_legged_rl/
    ├── README.md          詳しい使い方（観測・行動・報酬・ランダム化の設定、各 Phase の手順）
    ├── docker/            Isaac Sim 5.1 + Isaac Lab v2.3.0 のコンテナ
    ├── docs/              実装計画（IMPLEMENTATION_PLAN.md）、開発ログと評価結果（DEV_LOG.md）
    ├── scripts/           学習・評価・再生・方策の出力・URDF の検査
    ├── source/wheel_legged_rl/wheel_legged_rl/
    │   ├── configs/go2w/  Phase ごとの設定（報酬の重み、指令範囲、地形、ランダム化、評価基準）
    │   ├── envs/ rewards/ observations/ curriculum/ terrains/ events/ actuators/ ...
    │   └── deployment/    実機用の deploy.yaml と ONNX の出力、推論インターフェース
    └── tests/             Isaac Sim なしで実行できる単体テスト（pytest）
```

## 動かし方

必要な環境:

- NVIDIA GPU（RTX 3060 12 GB で動作を確認）
- Docker と NVIDIA Container Toolkit

Isaac Sim 5.1.0 と Isaac Lab v2.3.0 はコンテナに含まれています。

```bash
cd wheel_legged_rl
./scripts/fetch_go2w_description.sh      # Go2-W の URDF とメッシュを取得（このリポジトリには含めていません）
cd docker
./setup_host.sh                          # 初回のみ
cp .env.example .env                     # W&B を使う場合は WANDB_API_KEY を記入
docker compose build
cd ..

# 学習（Phase 1 → 4。Phase 3 以降は、前の Phase の best.pt を自動で読み込みます）
./docker/wl.sh train.py --task WL-Go2W-Phase1-Flat-v0 --headless
./docker/wl.sh evaluate.py --task WL-Go2W-Phase1-Flat-Play-v0 --load_run <run> --all_checkpoints --every 5 --select_best --headless
./docker/wl.sh train.py --task WL-Go2W-Phase2-Flat3Axis-v0 --headless --checkpoint logs/rsl_rl/go2w_wheel_leg/<phase1 run>/best.pt
./docker/wl.sh train.py --task WL-Go2W-Phase3-FlatCoord-v0 --headless
./docker/wl.sh train.py --task WL-Go2W-Phase4-Steps-v0 --headless

# 評価と再生
./docker/wl.sh evaluate.py --task WL-Go2W-Phase4-Steps-Play-v0 --load_run <phase4 run> --all_checkpoints --every 5 --select_best --headless
./docker/wl.sh play.py --task WL-Go2W-Phase4-Steps-Play-v0 --load_run <phase4 run> --checkpoint best.pt

# Isaac Sim なしの単体テスト
./docker/wl.sh -m pytest
```

各 Phase で 3000〜4000 iteration を学習します（Phase 4 は RTX 3060 で約 9 時間）。Phase 2 以降も、評価で `--select_best` を実行して best.pt を作ってから、次の Phase に進んでください。詳細な手順と設定は [wheel_legged_rl/README.md](wheel_legged_rl/README.md) を参照してください。

## 使用した主なソフトウェアとライセンス

- Isaac Sim 5.1.0、Isaac Lab v2.3.0、rsl-rl-lib 3.0.1、PyTorch
- Go2-W の URDF とメッシュ（Unitree Robotics、BSD-3-Clause）は、`scripts/fetch_go2w_description.sh` で `unitree_ros` から取得します。このリポジトリには含めていません。
- `scripts/cli_args.py` は Isaac Lab（BSD-3-Clause）のコードを基にしています。
- アクチュエータの設定と deploy.yaml の形式は、`unitree_rl_lab` を参考にしました。
