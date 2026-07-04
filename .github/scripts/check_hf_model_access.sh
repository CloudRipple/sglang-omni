#!/usr/bin/env bash
# Preflight: verify the configured HF token can read every CI model repo.
#
# Runs during environment setup, before any weight download, so a missing or
# mis-scoped HF_TOKEN fails fast with a consolidated report instead of a cryptic
# 404 surfacing deep inside a worker process at test time.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <venv-name> <hf-repo-id> [<hf-repo-id> ...]" >&2
  exit 1
fi

VENV_NAME="$1"
shift

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"

source "${VENV_NAME}/bin/activate"

python - "$@" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
api = HfApi(endpoint=os.environ.get("HF_ENDPOINT"), token=token)

failures: list[tuple[str, str]] = []
for repo_id in sys.argv[1:]:
    if Path(repo_id).is_dir():
        print(f"OK   local path: {repo_id}")
        continue
    try:
        info = api.model_info(repo_id)
    except GatedRepoError as exc:
        failures.append((repo_id, f"gated repo, token not granted access ({exc})"))
    except RepositoryNotFoundError:
        failures.append(
            (repo_id, "404 (repo missing, or private and this token lacks access)")
        )
    except Exception as exc:
        failures.append((repo_id, f"{type(exc).__name__}: {exc}"))
    else:
        visibility = "private" if info.private else "public"
        print(f"OK   {visibility:<7} {repo_id}")

if failures:
    print("\nHF access preflight FAILED for the following model repo(s):", file=sys.stderr)
    for repo_id, reason in failures:
        print(f"  - {repo_id}: {reason}", file=sys.stderr)
    print(
        "\nThe configured HF_TOKEN secret cannot access the repo(s) above. Update "
        "the repository secret 'HF_TOKEN' to a token with read access to every "
        "model, or grant this token access to the repo(s) on HuggingFace.",
        file=sys.stderr,
    )
    raise SystemExit(1)

print("\nAll CI model repos are accessible with the configured HF token")
PY
