# 開発ログ

## 2026-09-29 — Phase 4（小段差）の学習結果

### 背景

`2026-09-28_10-52-37_phase4_steps`（3000 iteration、Phase 3 の best.pt = `2026-09-27_01-26-37_phase3_flatcoord/model_2500.pt` から初期化）を評価しました。地形・地形カリキュラム・評価シナリオは README の Phase 4 の節と `configs/go2w/phase4_steps.yaml` のとおりです。評価は `WL-Go2W-Phase4-Steps-Play-v0` で 500 iteration ごとの checkpoint に対して行い、条件は 64 台 × 21 シナリオ（平地 15、段差 6）です。Play タスクなので物理のランダム化は有効で、観測ノイズと push は無効です。出力は `<run>/eval/2026-09-29_11-58-03/` です。

### 結果（best = model_2500、score −0.827）

完了条件をすべて満たしました。model_1000 以降の checkpoint は、すべて完了条件を満たしています。

| 指標 | model_0（= Phase 3 の best） | model_2500 | 基準 |
|---|---|---|---|
| 平地の転倒率 | 0.1% | 0% | < 1% |
| 平地の xy 速度 RMSE [m/s] | 0.081 | 0.073 | < 0.15 |
| 平地の yaw rate RMSE [rad/s] | 0.070 | 0.067 | < 0.2 |
| 平地の静止時の残留速度 [m/s] | 0.012 | 0.009 | < 0.05 |
| 平地の姿勢 RMS [rad] | 0.034 | 0.032 | < 0.05 |
| 段差の踏破率 | 43.8% | 100% | ≥ 80% |
| 段差の転倒率 | 6.0% | 0% | – |

シナリオごとの段差の踏破率:

| シナリオ | model_0 | model_500 | model_1000 | model_1500 以降 |
|---|---|---|---|---|
| curbs_up_0.10 | 0% | 55% | 100% | 100% |
| curbs_up_0.15 | 0% | 5% | 89% | 100% |
| curbs_up_back_0.15 | 0% | 50% | 100% | 100% |
| curbs_down_0.10 / 0.15 | 100% | 100% | 100% | 100% |
| random_grid_0.075 | 62% | 100% | 100% | 100% |

- Phase 3 の方策は、下りの縁石は越えられますが、上りの縁石は 0.10 m でも越えられませんでした（段の手前で止まり、段差シナリオの vx_bias は −0.476 m/s）。上りの段は車輪半径 0.086 m より高いので、脚を使う必要があります。Phase 4 の学習で、1000 iteration までに上りの縁石を越えられるようになり、1500 iteration 以降は 6 シナリオすべてで 100% になりました。
- 平地の 5 指標は model_0 と同程度か、わずかに良くなっています。平地を 20% 残したことで、忘却は見られません。
- 段差シナリオの姿勢 RMS は 0.076 rad（平地は 0.032）で、xy 速度 RMSE は 0.117 m/s です。段を越えるときに車体が傾き、速度が乱れるためです。基準は平地のシナリオだけに適用しています。
- 高速直進の速度不足（vx_bias [m/s]、model_2500）: forward_2.0 は −0.107（Phase 3 の評価では −0.156）、go_stop_2.0 は −0.110、backward_2.0 は −0.146（同 −0.115）。後退 2 m/s のシナリオ単体の xy 速度 RMSE は 0.148 で、基準 0.15 にほぼ並びます（基準は平地グループ全体に適用するので合格）。前進は改善しましたが、後退の不足は残っています。評価地形が Phase 3 と異なるので、Phase 3 との比較は参考値です。
- score は model_1500（−0.813）、model_2000（−0.812）、model_2500（−0.827）、model_2999（−0.817）で差が小さく、どれを best にしても完了条件は満たします。

### 注意・次の対応

- 段差は、学習で使った最大の高さ（0.15 m）と、奥行 1 m の縁石でしか評価していません。0.15 m を超える段や、奥行の浅い階段（0.3 m）は Phase 5 で扱います。
- 後退 2 m/s で約 7% の速度不足が残っています。Phase 3 の注意に挙げた対応（`mech_power_*` を弱める、追従の `std` を速度に応じて広げる、高速域の指令の比率を上げる）は、まだ行っていません。
- 評価は seed 1 つ、64 台です。学習中の `Curriculum/terrain_levels` などの推移は、この記録では確認していません。

## 2026-09-27 (2) — Phase 3 の学習結果と追加の評価

### 背景

`2026-09-27_01-26-37_phase3_flatcoord`（3000 iteration、Phase 2 の best.pt から初期化）を評価しました。完了条件は model_500 以降のすべての checkpoint で満たしています。一方で、学習が進むにつれて高速域の直進追従が悪化していました。そのため、次の 2 つを切り分ける評価を追加しました。

- Phase 2 と比べた悪化が、忘却によるものか、ランダム化によるものか
- 直進の誤差が、速度の不足なのか、超過なのか

### 結果（ランダム化あり、Phase 3 の評価、64 台 × 15 シナリオ）

best = model_2500（score 0.184）。

| 指標 | Phase 2 の best（= model_0） | model_2500 | 基準 |
|---|---|---|---|
| 転倒率 | 20.4% | 0% | < 1% |
| xy 速度 RMSE [m/s] | 0.183 | 0.080 | < 0.15 |
| yaw rate RMSE [rad/s] | 0.201 | 0.067 | < 0.2 |
| 静止時の残留速度 [m/s] | 0.035 | 0.006 | < 0.05 |
| 姿勢 RMS [rad] | 0.095 | 0.033 | < 0.05 |

±2.0 m/s のシナリオでは、転倒率が 38〜91% から 0% になりました。横移動の車輪スリップ率は 0.81〜0.91（Phase 2）から 0.44〜0.49 に下がり、`wheel_slip` の効果が出ています。その場旋回の CoT は約 3、スリップ率は約 0.7 です。Phase 2 でも同程度で、車輪を横に滑らせて曲がるスキッドステアの性質によるものと考えています。

### 変更内容

- **評価指標** `vx_bias` / `yaw_rate_bias`（`evaluation/metrics.py`）: 指令方向を正とした誤差の平均で、負の値は指令より遅いことを表します。|指令| < 0.1 のサンプルは除外し、指令が常に 0 のシナリオでは NaN になります。
- **`configs/go2w/phase3_flatcoord_nominal_eval.yaml`**（評価専用）: Phase 3 を `_base_` に、モデルの条件を Phase 2 に戻した設定です。条件は、摩擦 0.8 / 0.6、base 質量 ±1 kg、遅延 0、その他のランダム化なしです。critic の特権観測は公称値として残るので、Phase 3 の checkpoint をそのまま読み込めます。

### 確認

- pytest 163 件 pass、ruff は問題なし。
- 出力: `<run>/eval/nominal/`（Phase 2 の条件）と `<run>/eval/randomized_bias/`（ランダム化あり、bias 付きで再評価）。`--select_best` は付けていないので、best.pt（model_2500）は変わっていません。ランダム化ありの再評価は、元の評価と score が一致しました（seed 0 で再現）。
- **Phase 2 の条件で比較**: model_0 は、共通の 11 シナリオで Phase 2 の best.json とほぼ同じ値になりました（xy 速度 RMSE の差は 0.003 以内）。この設定で Phase 2 と同じ条件を再現できています。
  - model_500 以降の各 checkpoint は、共通シナリオすべてで基準を満たしています。
  - 前進と go_stop の xy 速度 RMSE は Phase 2 と同程度（0.02〜0.08）で、後退は 0.044 から 0.04〜0.07 に少し増えました。姿勢 RMS は model_1000 以降の全体で 0.017〜0.023 で、Phase 2（0.024）より良くなっています。
  - 横移動・旋回・円弧・斜め・turn_stop の xy 速度 RMSE は、0.02〜0.04 から 0.03〜0.09 に増えました（基準 0.15 の範囲内）。大きな忘却はありません。
  - この条件での最良は model_1500 です（score 0.127、全体の xy 速度 RMSE 0.048）。
- **直進の速度不足**（vx_bias [m/s]、ランダム化あり / Phase 2 の条件）:

  | シナリオ | model_1500 | model_2500 | model_2999 |
  |---|---|---|---|
  | forward_1.0 | −0.015 / +0.009 | −0.072 / −0.076 | −0.053 / −0.036 |
  | forward_2.0 | −0.088 / −0.057 | −0.156 / −0.133 | −0.180 / −0.147 |
  | go_stop_2.0 | −0.097 / −0.063 | −0.161 / −0.135 | −0.177 / −0.144 |
  | backward_2.0 | −0.077 / −0.063 | −0.115 / −0.112 | −0.107 / −0.093 |

  直進の RMSE は、ほとんどがこの偏りです。速度不足は、ランダム化なしでもほぼ同じだけ生じています。model_1500 から後で不足が大きくなり、同じ期間に 1 m あたりのエネルギーが減っています（Phase 2 の条件の forward_1.0 で 24.8 → 16.7 J/m）。エネルギーの罰則（`mech_power_*`、`joint_torques_legs`）が、追従と引き換えに速度を落とさせていると考えられます。円弧・斜めの vx_bias は ±0.05 以内で、偏りはありません。
  - yaw rate の偏りは、学習とともに小さくなっています（ランダム化ありの全体で −0.078 → −0.022）。

### 注意・次の対応

- 2 m/s で約 8% 遅い（model_2500）のは、基準（0.15 m/s）の範囲内ですが、学習を続けると悪化する傾向です。次に Phase 3 を学習し直す場合、または Phase 4 で次の対応を検討してください。
  - `mech_power_*` を弱める
  - 追従報酬の `std` を速度に応じて広げる
  - 高速域の指令の比率を上げる
- best の選び方で結果が変わります。ランダム化ありでは model_2500、Phase 2 の条件では model_1500 が最良でした。速度追従を重視する場合は、model_1500 を Phase 4 の初期値にすることも検討してください（ランダム化ありの score は 0.215 で、2500 の 0.184 より悪い）。

## 2026-09-27 — Phase 3 の critic に特権情報を追加し、critic の warm-up を導入

### 背景

Phase 3 のドメインランダム化の値を critic だけに与える、非対称な actor-critic にしました。Phase 2 の best.pt は actor・critic ともできるだけ引き継ぎ、学習開始直後は actor を凍結して critic だけを学習します。遅延は 1 台につき 1 つの値を全アクチュエータ群に適用する前回の実装のままです（ユーザーが了承済み）。

### 変更内容

- **関節摩擦のランダム化**（`events.randomize_joint_friction`、reset）: Go2-W の spec では関節摩擦の公称値が 0 なので、倍率ではなく絶対値でランダム化します。クーロン 0〜0.2 N·m、粘性 0〜0.02 N·m·s/rad、モータごとの推定値です。`QDDActuator` の摩擦テンソルを `friction_static` / `friction_viscous` として公開しました。
- **critic の特権観測**（`observations/privileged.py`、90 次元）: 車輪 4 つの静止・動摩擦 8、base の追加質量・重心オフセット xyz・慣性スケール 5、Kp スケール（脚）12、Kd スケール 16、モータ強度 16、関節摩擦 32、遅延 1。すべて公称値が 0、ランダム化の範囲がおおよそ ±1 になるよう正規化しています。接地摩擦は shape ごとにサンプルされる（4 輪で値が異なる）ので、車輪ごとに渡します。startup で決まる値（摩擦・質量・重心・慣性）は PhysX から読んでキャッシュします。startup イベントは観測マネージャの生成後に走り、マネージャは生成時に各項の `reset()` を呼ぶので、`WheelLeggedEnv.__init__` の最後で明示的に `refresh()` します（最初は「最初の reset で読み直す」実装にしましたが、ランダム化前の値が残っていたため直しました）。既存の critic 項の後ろに追加しているので、先頭の 263 列の意味は変わりません。Phase 1 / 2 の YAML では `null` です。
- **checkpoint の拡張**（`agents/critic_warmup.expand_state_dict`、`WheelLeggedOnPolicyRunner.load_init`）: shape が同じ項はそのまま読み込みます。critic の第 1 層の重みは、入力が増えた場合だけ、既存の列をコピーして新しい列を 0 にします。それ以外の不一致はエラーです。optimizer は読み込みません（再初期化）。
- **warm-up**（`agents/runner.py`、`CriticWarmupCfg`）: actor の MLP と action std を `requires_grad=False` にし、更新の前後で actor のパラメータが変わっていないことを毎回 assert します。warm-up 中は `schedule="fixed"`、学習率 3e-4、epoch 数 2 です。解除時に adaptive、1e-3、5 epoch に戻します。
- **解除の判定**（`CriticWarmup`）: 毎 iteration、更新前に新しい rollout で次の値を計算します。
  - 更新前 EV
  - value MSE
  - 相対バイアス mean(R − V) / std(R)
  - 前回の更新後 EV との差（ev_gap）
  
  critic だけの更新を最低 2 回、最大 20 回行い、その間に「EV が頭打ち（上昇 < 0.01 が 2 回続く）かつ EV ≥ 0.5 かつ |相対バイアス| ≤ 0.1」か「ev_gap > 0.1」になったら解除します。
- **タイマーの分散**（`WheelLeggedEnv.desynchronize_timers`、train.py）: 学習開始時に、指令の再サンプル（10 s）と push（10〜15 s）のタイマーを周期内の一様乱数にします。
- **ログ**: `Critic/*` を毎 iteration、`Warmup/*` を warm-up の設定がある run で出します。

### 確認

- pytest 162 件 pass、ruff は問題なし。
- 4096 env のスモークテストで、Phase 2 の best.pt から `critic.0.weight: (512, 263) -> (512, 353)` に拡張して読み込み、warm-up を経て通常の PPO に移ることを確認しました。
- 解除条件を無効にして、critic だけの更新を 20〜26 回行ったときの推移を記録しました。既定値はこの結果から決めています。
  - **学習率**: 1e-3（PPO と同じ）では、新しく作った Adam の最初の更新で critic が崩れました（同じバッチでの EV が 0.70 → 0.26）。3e-4 では崩れず（0.70 → 0.77）、最終的な EV は同程度（約 0.93〜0.95）でした。そのため warm-up の学習率を 3e-4 にしました。
  - **ev_gap**: 健全な warm-up でも、iteration ごとのばらつきで 0〜0.05 になりました。0.05 では誤って解除されたので、0.1 にしました。
  - **タイマー**: 分散する前は、21 iteration 目（約 10 s）に更新前 EV が 0.95 → 0.68 に急落しました。全 env の指令変更と最初の push が同時に起きるためで、warm-up（最大 20 iteration）のデータには push も指令変更もほとんど含まれていませんでした。分散した後は急落が消え、value MSE は最初から約 0.02 で安定しました。
  - **既定の設定での結果**: 8 回の更新後に `converged`（EV 0.89、相対バイアス −0.08）で解除されました。
- 512 env で、特権観測の 7 項すべてが [-1, 1] に収まり、同じロボットでも車輪ごとに摩擦が違うことを確認しました。Phase 2 の critic は 263 次元のままです。
- 上の確認値はすべてスモークテスト（最大 30 iteration）のものです。本番の学習（3000 iteration）はまだ行っていません。

### 注意

- 解除直後の最初の更新で KL が大きくなり、adaptive LR が下限の 1e-5 まで下がります。その後 3 iteration ほどで約 1.7e-4 に戻ります。Phase 2 の開始時（optimizer 再初期化）にも同じ挙動がありました。解除時の学習率を 1e-4 から始めても変わらなかったので、そのままにしています。
- 関節摩擦の範囲は推定です。実機で測定できたら置き換えてください。
- Phase 3 で DR の項を Hydra で `null` にしても、特権観測は残ります（公称値に対応する値が入ります）。

## 2026-09-26 (2) — Phase 3（平地・高速域・Sim-to-Real のドメインランダム化）を追加

### 背景

Phase 2 の `2026-09-26_04-26-55_phase2_flat3axis` は評価の完了条件をすべて満たしました（best = model_2500: xy 速度 RMSE 0.035 m/s、yaw rate RMSE 0.037 rad/s、転倒率 0）。計画 §12 の Phase 3（摩擦の乱択、高速域、エネルギー報酬）に、Sim-to-Real で優先度が高いモデル誤差のランダム化（指令遅延、モータ強度、Kp/Kd、質量・重心・慣性）を加えました。実機へのデプロイはまだ先なので、地形を扱う Phase 4 より前に入れる方針です。

### 変更内容

- タスク `WL-Go2W-Phase3-FlatCoord-v0` / `-Play-v0`、`configs/go2w/phase3_flatcoord.yaml`、`Go2WPhase3PPORunnerCfg`（run 名 `phase3_flatcoord`、3000 iteration）。
- 初期化: runner 設定に `init_from` を追加しました。Phase 3 は `"phase2_flat3axis"` で、`train.py` は最新の該当 run の `best.pt` から学習を始めます。その run に `best.pt` がなければエラーになります（古い run へ黙って戻ることはしません）。`--init_from` は run ディレクトリ（→ `best.pt`）も受け付けます。
- 指令: vx の上限を ±1.0 → ±2.0 m/s に広げました（±1.0 から delta 0.1 のカリキュラム）。vy・ωz は最初から Phase 2 の上限です。速度が上がって 1 エピソードの走行距離が最大 40 m になるため、地形の `border_width` を 50 m にしました。
- 報酬: Phase 2 の項に `mech_power_legs` −2e-4、`mech_power_wheels` −1e-4、`joint_torques_legs` −2e-5、`wheel_slip` −0.1 を追加しました。いずれも 1 m/s 走行時に追従報酬の 1〜3% 程度になるよう選んだ初期値です。
- ドメインランダム化（`events/events.py` と `EventCfg`）:
  - 指令遅延 0〜4 physics step（0〜20 ms）。`DelayedPDActuator` はアクチュエータ群ごとに独立して遅延をサンプルするため、`sync_actuator_delay`（reset イベント）で 1 台につき 1 つの遅延を全群に適用します。実機では全モータの指令が 1 つのパケットで送られるためです。
  - モータ強度 ×0.8〜1.2: `QDDActuator.strength` を追加し、`s * clip_TN(τ)` としました。トルク定数の誤差として、指令トルクと T-N 曲線のトルク軸を同時にスケールします（速度軸はそのまま）。
  - Kp / Kd ×0.8〜1.2: Isaac Lab の `randomize_actuator_gains`。車輪は Kp = 0 なので Kd だけが変わります。
  - base 質量 −1〜+3 kg（Phase 1/2 は ±1 kg）、脚・車輪リンクの質量 ×0.9〜1.1（いずれも慣性を質量比で変更）、base の慣性 ×0.8〜1.2（一様スケールなので慣性テンソルとして有効なまま）、base の重心 x ±3 cm・y/z ±2 cm。
  - 摩擦 0.3〜1.2（計画どおり）。
- Phase 1 / 2 の YAML では、新しい項目を `null` にして無効化しました（`actuators` の指定がないので遅延は 0）。
- 評価: Phase 1/2 のシナリオに、±2.0 m/s、2.0 m/s からの停止、円弧（1.5 m/s・1.0 rad/s）を追加しました。完了条件は Phase 2 の条件に「姿勢 RMS（roll・pitch）< 0.05 rad」を加えたものです。Play タスクでもランダム化は有効なので、評価はランダム化されたロボットで行われます。
- 学習ログと評価の輸送コスト（CoT）の質量を、env 0 の値から全 env の平均に変えました。

### 確認

- pytest 144 件 pass、ruff は問題なし（コンテナ内）。
- 64 env × 3 iteration のスモークテスト: `--init_from` を指定せずに、Phase 2 の `best.pt` から初期化されること（action std 約 0.22）、報酬 15 項とイベント（startup 5 項、reset 5 項）が登録されることを確認しました。
- 256 env で、値が PhysX とアクチュエータに反映されていることを確認しました。Phase 3 では、base 質量の差が −0.99〜+3.00 kg、脚リンクの質量が ×0.90〜1.10、慣性が ×0.80〜1.20、Kp/Kd/強度が ×0.80〜1.20、遅延が 0〜4 step（全群で同一）、摩擦が 0.31〜1.20 でした。Phase 2 では、これらはすべて従来どおりの値のままでした。
- ベースライン: Phase 2 の best.pt を Phase 3 の評価（32 env）にかけると、全体の転倒率は 18.5% でした。±2.0 m/s では 31〜81%、ランダム化のみで既存の指令範囲でも、旋回 ±1.5 rad/s で 12〜19%、横移動で 3〜9% 転倒しました（`logs/eval_phase2best_on_phase3/`）。

### 注意

- すべてのランダム化を一度に有効にしています。学習の序盤に崩れる場合は、Hydra で範囲を狭めて（例: `env.events.motor_strength.params.strength_range=[0.9,1.1]`）学習し、次の run で広げてください。遅延の範囲は 3 つのアクチュエータ群で同じにする必要があります（違うと `sync_actuator_delay` がエラーを出します）。
- critic に特権情報（摩擦・追加質量・遅延）は追加していません。critic の入力次元が変わり、Phase 2 の critic を読み込めなくなるためです。→ 次の項目（2026-09-27）で、critic の重みを拡張して追加しました。
- 報酬の重みは初期値です。2 m/s での追従が安定してから、W&B の `Diagnostics/mech_power_*` を見て強めてください。

## 2026-09-26 — Phase 2（平地・3 軸指令）を追加

### 背景

Phase 1 の `2026-09-25_12-05-14_phase1_flat` は評価の完了条件をすべて満たしました（best = model_2500: vx RMSE 0.017 m/s、転倒率 0、静止時の残留速度 0.004 m/s）。計画 §12 に従い、Phase 2 を登録します（方針 1 の「実機確認後」は、ユーザーの指示により前倒し）。

### 変更内容

- タスク `WL-Go2W-Phase2-Flat3Axis-v0` / `-Play-v0`、`configs/go2w/phase2_flat3axis.yaml`、`Go2WPhase2PPORunnerCfg`（run 名 `phase2_flat3axis`、3000 iteration）。
- 指令: vx は最初から ±1.0（Phase 1 の分布を維持して忘却を防ぐ）。vy は ±0.1 → ±0.5（delta 0.05）、ωz は ±0.3 → ±1.5 rad/s（delta 0.1）をカリキュラムで拡大。静止 20%。静止ゲートは使いません。
- 報酬: Phase 1 の項に `joint_pos_limits_legs` −10、`undesired_contacts` −1 を追加。`action_rate` −0.01 と `stand_still` −0.5 は Phase 1 から継続。
- 評価: Phase 1 のシナリオ（stand / ±1.0 m/s / go_stop）に、横移動 ±0.5、旋回 ±1.5、円弧、斜め、turn_stop を追加。完了条件は xy 速度 RMSE < 0.15 m/s、yaw rate RMSE < 0.2 rad/s、転倒率 < 1%、静止時の残留速度 < 0.05 m/s。
- `train.py`: `--resume` なしの `--checkpoint <ファイル>` を `--init_from` として扱います。ホストのパスはコンテナ内のパス（`/quadra/...`）に読み替えます（`_paths.resolve_project_file`）。

### 確認

- pytest 138 件 pass（コンテナ内）。
- 64 env × 3 iteration のスモークテスト（Phase 1 の model_3999.pt から初期化、ホストパス指定）: 完走。action std が約 0.12（初期値 1.0 ではない）で、重みが引き継がれたことを確認。報酬の重みとカリキュラムの設定が env.yaml で意図どおり。

### 注意

- 車輪の回転軸は y 方向なので、横方向に転がることはできません。vy は脚を使った横移動（ステップ、またはすり足）で実現する必要があり、Phase 2 でいちばん難しい軸です。`Curriculum/lin_vel_y_levels/range` が広がらない場合は、`limit_ranges.lin_vel_y` を下げるか、脚の接地時間の報酬（計画上は Phase 5）を前倒しで検討してください。
- vy のカリキュラムは xy の追従報酬で判定するため、vx の追従が良いと、vy の追従が悪くても範囲が広がることがあります。評価の `left_0.5` / `right_0.5` で確認してください。

## 2026-09-25 — Phase 1 を「静止 → 走行」の 2 段階カリキュラムに変更

### 背景

`WL-Go2W-Phase1-Flat-v0` の最初の学習では、胴体を低く沈めたまま走行する方策になりました。Phase 1 の報酬には高さの項も静止の項もなく、姿勢を下げても罰則がなかったためです。

### 変更内容

- **段階 1(静止)**: 初期の指令範囲を `lin_vel_x ∈ [−0.027, 0.027]` m/s とし、すべての指令を 0.1 km/h(0.0278 m/s)未満にしました。`stand_still`(−0.5、`command_threshold` 0.0277 m/s)と `base_height`(−30、目標 0.408 m)を有効にしました。
- **段階 2(走行)**: `command_levels` に `stand_gate` を追加しました。ゲートが開くまでは指令範囲を初期値に固定します。ゲートが開いた後は従来どおり、追従報酬が最大値の 80% を超えるたびに ±0.1 m/s ずつ ±1.0 m/s まで広げます。`base_height` は段階 2 でも有効です。
- **ゲートの条件**: 前回の判定以降に終わったすべてのエピソードで集計し、次の条件を 3 回連続で満たしたら開きます。判定は 1 エピソード長ごとです。
  - time-out で終わった割合 ≥ 0.95(転倒 ≤ 5%)
  - `stand_still` の 1 ステップ平均の生の値 ≤ 0.5
  - `base_height` の 1 ステップ平均 ≤ 0.001 m²(RMS 約 3 cm)
- **ログ**: `Curriculum/lin_vel_x_levels/{stand_stage, range, time_out_ratio, stand_still_cost, base_height_cost}`。従来の `Curriculum/lin_vel_x_levels` は `.../range` に変わりました。

### 確認

- pytest 136 件 pass、ruff は問題なし。
- 64 env、`episode_length_s=1.0` のスモークテスト: ゲートが閉じている間は範囲が 0.054 のまま固定されました。条件を緩めた実行では `stand_stage` が 1 から 0 になり、範囲が 0.254 に広がりました。

### 注意

- 段階は checkpoint に保存されません。`--resume` で再開すると段階 1 から始まります。すでに静止できる方策なら、3 回の判定の後に段階 2 へ進みます。
- 閾値と重みは初期値です。実際の学習でゲートが開かない場合や、開くのが早すぎる場合は、wandb の `*_cost` を見て調整してください。

## 2026-09-24 (2) — Docker 上の Isaac Sim 5.1 で初回動作確認

### 実行環境

`docker/` に Isaac Sim 5.1.0 + Isaac Lab v2.3.0 のコンテナを追加しました。`ur10-robotiq-grasp` の Docker 構成（NGC 公式イメージをベースに、compose で GPU・X11・キャッシュ用ボリュームを設定し、リポジトリをマウント）に合わせています。

- ベースイメージ: `nvcr.io/nvidia/isaac-lab:2.3.0`
- 追加パッケージ: rsl-rl-lib 3.0.1、onnx 1.19.1、onnxruntime 1.30.0、wandb 0.22.2
- 同梱のバージョン: isaaclab 0.47.1、torch 2.7.0+cu128
- ホスト: RTX 3060 12 GB、driver 580.173.02
- イメージは Isaac Sim 5.1.0-rc.19 のビルドです。
- `docker/wl.sh <script>` で `scripts/` 内のスクリプトをコンテナ内で実行できます。

### 結果

| 手順 | 結果 |
|---|---|
| コンテナ内 pytest（Isaac Sim 同梱の Python 3.11） | 135 件すべて pass |
| `joint_test.py`（inspect / joints / stand / roll） | **31/31 PASS**（`logs/joint_test.json`） |
| 学習の煙テスト（64 env × 5 iteration） | 完走。`deploy.yaml` は obs 240 / action 16。報酬の削除と Diagnostics の記録も確認 |
| `export_policy.py` と `PolicyInterface`（実際の checkpoint、Isaac 外） | 出力と推論の両方が成功 |
| `play.py --export_only` | TorchScript / ONNX を出力 |
| `evaluate.py --all_checkpoints --select_best` | metrics.json / csv と best.pt を出力。5 iteration なので完了条件は未達（想定どおり）で、終了コードは 1 |

実機との対応に関わる確認結果:

- sim の関節順は幅優先で `FL,FR,RL,RR × hip,thigh,calf` の次に車輪 4 つです。脚が 0..11、車輪が 12..15 になり、Unitree の C++ deploy の前提を満たします。`joint_ids_map` は `[3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8, 13, 12, 15, 14]` で、単体テストの予測と一致しました。
- 16 関節すべてで、回転方向が URDF の軸と一致しました（cos = +1.000）。車輪の正転（+y 軸まわり）で前進することも確認しました。
- 転がりの有効半径は、円柱タイヤで 0.0853 m（仕様 0.0860 m に対して −0.8%）、元のメッシュ（convex hull）で 0.0832 m（−3.3%）でした。+5 rad/s を指令したときの到達速度は、円柱が 4.27 rad/s、メッシュが 3.98 rad/s です。**円柱タイヤを採用**します。

### Isaac Sim 上で見つかり、修正した問題

1. **車輪の速度ループが数値的に発散していた（重要）。** 車輪単体の回転慣性は 0.0012 kg·m² しかなく、Kd 0.5 を explicit に計算すると Kd·dt/I = 2.1 となり、安定限界の 2 を超えます。+5 rad/s の指令に対して +21 / −17 rad/s になっていました。車輪に armature 0.005 kg·m²（推定の反射ロータ慣性）を追加し、比を 0.37 に下げました。あわせて `urdf_audit.py` に、関節ごとの Kd·dt/(I + armature) < 1 を確認するチェックを加え、単体テストで回帰を防いでいます。計画書・unitree_rl_lab ともに armature を設定していなかった点です。
2. **Isaac Sim 5.1 の URDF インポータは、質量を持つ fixed リンクをマージしない。** `*_foot_motor`、`Head_upper`、`Head_lower` は別のボディとして残り、マージされるのは質量のない `calflower` だけでした。対応として次の 2 点を行いました。
   - 頭から地面に接触しても終了しないため、`Head_.*` を終了条件と接触ペナルティの対象に加えました。
   - `joint_test` の比較用参照を、実際に残ったボディに合わせるようにしました（`merged_bodies(keep=...)`）。
3. **終了処理の問題。** headless の `SimulationApp.close()` が、joint_test の後で止まることがありました。また evaluate では、close がプロセスを終了コード 0 で終わらせていました。どちらも結果を書き出したあと `os._exit` で終了するように変えました。
4. **`evaluate.py` のリセット。** inference mode の外で `env.reset()` を呼ぶと、inference tensor への in-place 書き込みで失敗していました。シナリオ全体を inference mode で実行するように直しました。
5. **joint_test の判定方法。**
   - 回転軸の一致は、関節の瞬時速度の符号で判定するようにしました。
   - 関節の追従は、Kp 25 と重力による定常偏差を許容し、ステップ量の 50% 以上で合格としました。
   - 立位の高さは、固定の公称値ではなく、実測の関節角から計算した FK（タイヤの軸方向オフセットとキャンバーを考慮）と比べるようにしました。

### 所見（未対応）

- FixStand 相当のゲイン（脚 Kp 80）で車輪を位置保持しても、自重で hip が外側に約 0.2 rad 開き、右側の hip トルクは 23.4 N·m で飽和しました。車体高さは 0.368 m でした。本番の Kp 25 で学習する方策が、この荷重をどう支えるか（外開きを抑えられるか）を Phase 1 の学習で確認する必要があります。
- 本学習（4096 env）はまだ実行していません。W&B を使う場合は、`docker/.env` に `WANDB_API_KEY` を設定してください。

## 2026-09-24 — 骨格 + Phase 1 の実装

### 実施内容

[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) §6 のファイル構成で、骨格と Phase 1 を実装しました。`randomization/`（計画上 Step 9）は、まだ作成していません。

### 検証結果（Isaac Sim 上では未実行）

| 検証 | 結果 |
|---|---|
| `pytest`（Isaac 不要、132 件。Go2-W 実 URDF を使うものを含む） | 全件 pass（Python 3.12 / torch 2.10 CPU / onnxruntime） |
| `check_isaaclab_api.py`（Isaac Lab `v2.3.0` と rsl-rl-lib `3.0.1` のソースに対する照合） | 619 参照、問題 0 件。誤った引数名・関数名・フィールド名を 3 つわざと入れて 3 件とも検出されることを確認済み |
| `urdf_audit.py --robot go2w` | RobotSpec との整合チェックはすべて pass（下記の数値を参照） |
| `ruff check` / `ruff format` | 問題なし |
| `export_policy.py`（rsl_rl 形式の模擬 checkpoint を使用） | TorchScript / ONNX の出力が元のネットワークと一致。ONNX の入力は `obs` [1, 240] の固定形状 |

URDF から再計算し、計画の値を確認したもの:

- 総質量 19.523 kg
- 車輪半径 0.0860 m、幅 0.0518 m、軸方向オフセット ±0.0481 m
- 立位高さ 0.4076 m（thigh 0.8、calf −1.5）
- fixed joint のマージ後は 17 ボディ。`*_foot_motor`（0.646 kg）は calf にマージされる

### 実行環境について（計画時の前提との違い）

計画時は「GPU / Isaac Sim なし」を前提にしていましたが、このマシンには RTX 3060（12 GB）があります。ただし Isaac Lab v2.3.0 の実行環境はないため、Isaac 上での実行（USD 変換・joint_test・学習）はしていません。

- `~/Documents/IsaacLab` は v2.1.1 で、`~/Documents/isaac_sim_env` の python は存在しない `/usr/bin/python3.10` へのシンボリックリンクになっており、壊れています。
- Docker イメージ（`polyos-isaac-lab`、`my-isaac-lab:*`）は Isaac Sim 6.0 rc / rsl-rl-lib 5.0.1 で、API の世代が異なります。

Isaac Sim 5.1 と Isaac Lab 2.3.0 を用意したら、README の「初回動作確認チェックリスト」を実行してください。

### 計画から変えた点と、実装時に決めた点

1. **配置**: リポジトリ直下の `wheel_legged_rl/` に、独立パッケージとして置きました。既存の `isaaclab/source/quadra`（旧プロトタイプ）には手を入れていません。
2. **Go2-W の description は同梱しない**: メッシュが約 30 MB あるため、`scripts/fetch_go2w_description.sh` で `unitree_ros@ccfc6fd` を取得する方式にしました。URDF が無い場合、テストは skip されます。
3. **車輪 collision**: `GO2W_SPEC.cylinder_wheels=True` を既定にしました。計画では「必要なら」差し替える方針でしたが、メッシュの convex hull では車輪が多角形になるためです。円柱は URDF のメッシュから半径・幅・オフセットを自動で合わせています。joint_test の roll で `False` の場合と比べ、結果をここに残してください。
4. **報酬の無効化**: YAML に書かれていない項目は重み 0 にし、Env の生成時に削除します。削除しても Hydra（`env.rewards.X.weight=...`）で再度有効にできます。`null` でも無効になります。
5. **Phase 1 の DR**: 計画 §12 では摩擦の乱択を Phase 3 からとしているため、Phase 1 は摩擦を固定（0.8 / 0.6）にしました。base 質量は ±1 kg、push は ±0.5 m/s（10〜15 s 間隔）、リセット時の関節オフセットは ±0.1 rad です。
6. **Phase 1 のコマンド**: vx は ±0.5 から始め、カリキュラム（追従報酬が最大値の 80% を超えたら ±0.1 ずつ拡大）で ±1.0 まで広げます。静止指令は 20% です。
7. **deploy.yaml の C++ 互換性**: `joint_ids_map`（sim 関節 i → SDK モータ番号）、SDK 順の Kp/Kd、actions と observations の形式は unitree_rl_lab の exporter と C++ ローダの実装に合わせました。一方、`terrain_height_scan` は Unitree の C++ に登録がないため、`policy.cpp_unsupported` に列挙しています。C++ で動かすには登録の追加が必要です。
8. **評価指標**: 指令を切り替えた直後の区間（`skip_s`）は、全シナリオで追従誤差と静止速度の計算から除外しています（`EpisodeLog.valid`）。
9. **診断の GPU 同期**: `Diagnostics/*` は 0 次元テンソルのまま rsl_rl に渡し、ステップごとの GPU→CPU 同期を避けています。例外は転倒率の計数だけです。

### 推定値（実機で確認が必要）

- calf の T-N 曲線: Go2HV × 1.5 と仮定（Y1 30.3、Y2 35.1、X1 9.0、X2 20.0）。URDF の effort 35.55 / velocity 20.07 とは整合します。
- 車輪の T-N 曲線: Go2HV と同一と仮定。Kd 0.5 は unitree_rl_lab の値です。
- 関節摩擦: 0。脚の armature: 0。Unitree は Go2HV についてこれらを公開していません。車輪の armature 0.005 kg·m² は数値安定のための推定値で、2026-09-24 (2) で追加しました。

### 未実装（計画上、後の Phase で扱うもの）

- Phase 2〜7 のタスク登録と YAML（方針 1 により、Phase 1 の実機確認後に追加します）
- `randomization/`（Step 9）、知覚ノイズの遅延（dropout と bias は `terrain_height_scan` の引数として実装済み）
- 地形種別ごとの `Diagnostics/<terrain>/vel_error`、複合地形（Phase 7）、低摩擦列のイベント、車輪の粘性抵抗項
- 段差・階段向けの脚用報酬（接地時間など）
