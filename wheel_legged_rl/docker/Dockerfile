# -----------------------------------------------------------------------------
# wheel_legged_rl 実行環境: Isaac Sim 5.1.0 + Isaac Lab v2.3.0 + rsl-rl-lib 3.0.1
#
# ur10-robotiq-grasp の Dockerfile と同じく NGC 公式の Isaac Lab イメージを土台にし、
# このプロジェクトに必要な追加パッケージだけを入れる。ソースはビルド時に取り込まず、
# docker-compose.yml でリポジトリごとマウントする（編集が即反映される）。
#
#   cd wheel_legged_rl/docker && docker compose build
# -----------------------------------------------------------------------------
ARG ISAACLAB_IMAGE=nvcr.io/nvidia/isaac-lab:2.3.0
FROM ${ISAACLAB_IMAGE}

USER root
WORKDIR /workspace/isaaclab

# git: fetch_go2w_description.sh / rsl_rl の add_git_repo_to_log、ffmpeg: --video
RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# rsl-rl-lib は Isaac Lab v2.3.0 の固定バージョン (3.0.1) に揃える。
# onnx: play.py の ONNX 出力、onnxruntime: PolicyInterface、wandb: --logger wandb（新形式 API キーには 0.22.3+）、pytest/ruff: 検証
RUN ./isaaclab.sh -p -m pip install --no-cache-dir \
        "rsl-rl-lib==3.0.1" onnx onnxruntime "wandb>=0.22.3" pytest ruff \
    && echo "Isaac Sim $(cat /isaac-sim/VERSION)" \
    && ./isaaclab.sh -p -c "import importlib.metadata as m; \
print({d: m.version(d) for d in ['isaaclab', 'rsl-rl-lib', 'torch', 'onnx', 'onnxruntime', 'wandb']})"

# マウントされた wheel_legged_rl を import 可能にする（scripts/_paths.py でも追加している）
ENV PYTHONPATH=/quadra/wheel_legged_rl/source/wheel_legged_rl
ENV WL_CACHE_DIR=/root/.cache/wheel_legged_rl

ENTRYPOINT ["/bin/bash"]
