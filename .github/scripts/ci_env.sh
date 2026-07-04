# Source CI-aligned env for all Omni benchmark tests (unit, Qwen3, TTS, Qwen3-ASR).
# Matches GitHub Actions omni-setup + tune-ci-thresholds auto_env.
umask 000
set -a
export OMNI_CI_HOME="${OMNI_CI_HOME:-/github/home/calibration}"
export HOME="${OMNI_CI_HOME}"
export USER="${USER:-sglang-omni}"
export LOGNAME="${LOGNAME:-sglang-omni}"
export HF_HOME=/github/home/.cache/huggingface
export MODELSCOPE_CACHE=/github/home/.cache/modelscope
export XDG_CACHE_HOME="${OMNI_CI_HOME}/.cache"
export CUDA_CACHE_PATH="${OMNI_CI_HOME}/.nv/ComputeCache"
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
export UV_CACHE_DIR=/github/home/uv-cache
# uv-managed CPython lives outside OMNI_CI_HOME (shared, never wiped by the
# per-PR `rm -rf ${OMNI_CI_HOME}` rebuild), so venvs stay linked to a live
# interpreter across stages/PRs.
export UV_PYTHON_INSTALL_DIR=/github/home/uv-python
export TORCHINDUCTOR_CACHE_DIR="${OMNI_CI_HOME}/.torchinductor"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export SEEDTTS_SIM_CACHE_DIR="${SEEDTTS_SIM_CACHE_DIR:-/github/home/seedtts-wavlm-sim}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
set +a
