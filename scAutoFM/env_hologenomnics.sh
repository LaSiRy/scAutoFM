# shellcheck shell=bash
# Local runtime helper: use conda env `hologenomnics` and run from this directory.
# Usage:
#   source ./env_hologenomnics.sh
#   python supernet_train_prompt.py --cfg=./experiments/scAutoFM/supernet/supernet-B_prompt.yaml ...

CONDA_ENV="${CONDA_ENV:-hologenomnics}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prefer the shared RSSD install; fall back to the path used by holo-genomic train scripts.
for _base in "${CONDA_BASE:-}" /app/rssd/sys/anaconda3 /home/biosan/anaconda3; do
  if [[ -n "${_base}" && -x "${_base}/envs/${CONDA_ENV}/bin/python" ]]; then
    CONDA_BASE="${_base}"
    break
  fi
done

ENV_BIN="${CONDA_BASE}/envs/${CONDA_ENV}/bin"
if [[ ! -x "${ENV_BIN}/python" ]]; then
  echo "ERROR: missing ${ENV_BIN}/python (conda env=${CONDA_ENV})" >&2
  return 1 2>/dev/null || exit 1
fi

export PATH="${ENV_BIN}:${PATH}"
export PYTHON="${ENV_BIN}/python"
# Avoid ~/.local site-packages shadowing the conda env (numpy mismatches).
export PYTHONNOUSERSITE=1
cd "${SCRIPT_DIR}" || return 1 2>/dev/null || exit 1
echo "scAutoFM runtime: env=${CONDA_ENV} python=${PYTHON} cwd=${PWD}"
