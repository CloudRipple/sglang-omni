#!/usr/bin/env bash
# Import probe for the Omni CI venv (matches packages exercised in real CI jobs).
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <venv-name>" >&2
  exit 1
fi

if [ -z "${OMNI_CI_HOME:-}" ]; then
  echo "OMNI_CI_HOME is not set" >&2
  exit 1
fi

VENV_NAME="$1"
PYTHON="${OMNI_CI_HOME}/${VENV_NAME}/bin/python"

if [ ! -x "${PYTHON}" ]; then
  echo "python not found: ${PYTHON}" >&2
  exit 1
fi

PROBE_LOG="$(mktemp)"
trap 'rm -f "${PROBE_LOG}"' EXIT

if ! "${PYTHON}" - <<'PY' >"${PROBE_LOG}" 2>&1; then
import importlib
import sys

print(f"python={sys.executable}")
print(f"prefix={sys.prefix}")

for module_name in ("av", "torch", "transformers", "sglang"):
    importlib.import_module(module_name)

normalizers = importlib.import_module("whisper.normalizers")
getattr(normalizers, "EnglishTextNormalizer")
PY
  echo "::error::${VENV_NAME} import probe failed at ${OMNI_CI_HOME}/${VENV_NAME}" >&2
  echo "----- import probe output -----" >&2
  cat "${PROBE_LOG}" >&2
  echo "----- end import probe output -----" >&2
  exit 1
fi

echo "Import probe ok: ${OMNI_CI_HOME}/${VENV_NAME}"
