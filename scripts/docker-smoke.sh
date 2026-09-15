#!/usr/bin/env bash
set -euo pipefail

curl --fail --silent --show-error http://127.0.0.1:8080/api/healthz \
  | python -c 'import json,sys; assert json.load(sys.stdin)=={"status":"ok","protocol":2}'
curl --fail --silent --show-error http://127.0.0.1:8080/ \
  | grep -q 'Fly Crossy Connectome'
python "$(dirname "$0")/websocket-smoke.py"
