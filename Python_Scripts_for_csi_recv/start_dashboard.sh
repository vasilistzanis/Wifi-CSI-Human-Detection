#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

ENV_FILE="web_app/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    printf 'Missing %s. Start with ./start_api.sh and configure it first.\n' "$ENV_FILE" >&2
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

CSI_EDGE_TOKEN="${CSI_EDGE_TOKEN:-}"
if [[ ${#CSI_EDGE_TOKEN} -lt 32 ]]; then
    printf 'CSI_EDGE_TOKEN in %s must contain at least 32 characters.\n' "$ENV_FILE" >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    printf 'Python environment not found. Create .venv and install dashboard dependencies.\n' >&2
    exit 1
fi

export CSI_SERVER_URL="${CSI_SERVER_URL:-http://127.0.0.1:8000}"
if ! curl --silent --fail --max-time 3 "$CSI_SERVER_URL/readyz" >/dev/null; then
    printf 'The API is not ready at %s. Start it in another terminal with ./start_api.sh.\n' \
        "$CSI_SERVER_URL" >&2
    exit 1
fi

exec .venv/bin/python live_dashboard.py \
    --port "${CSI_SERIAL_PORT:-/dev/ttyACM0}" \
    --model "${CSI_MODEL:-rf}" \
    "$@"
