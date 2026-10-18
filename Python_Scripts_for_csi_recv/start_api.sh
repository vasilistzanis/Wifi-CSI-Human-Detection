#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

ENV_FILE="web_app/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    printf 'Missing %s. Copy web_app/.env.example and configure it first.\n' "$ENV_FILE" >&2
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

CSI_EDGE_TOKEN="${CSI_EDGE_TOKEN:-}"
CSI_ADMIN_USERNAME="${CSI_ADMIN_USERNAME:-}"
if [[ ${#CSI_EDGE_TOKEN} -lt 32 ]]; then
    printf 'CSI_EDGE_TOKEN in %s must contain at least 32 characters.\n' "$ENV_FILE" >&2
    exit 1
fi
if [[ -z ${CSI_ADMIN_USERNAME:-} ]]; then
    printf 'CSI_ADMIN_USERNAME must be set in %s.\n' "$ENV_FILE" >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    printf 'Python environment not found. Create .venv and install web_app/requirements.txt.\n' >&2
    exit 1
fi

export CSI_DB_PATH="${CSI_DB_PATH:-$PROJECT_DIR/web_app/data/app.sqlite3}"
export CSI_COOKIE_SECURE="${CSI_COOKIE_SECURE:-false}"

if curl --silent --fail --max-time 2 http://127.0.0.1:8000/healthz >/dev/null; then
    printf 'The API is already running at http://127.0.0.1:8000.\n'
    exit 0
fi

exec .venv/bin/python -m uvicorn web_app.server:app --host 127.0.0.1 --port 8000
