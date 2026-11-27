#!/usr/bin/env bash
# Serve the IK2 monitor (Maya's Monitor page, IK2 data) on 0.0.0.0:8080.
set -euo pipefail
cd "$(dirname "$0")"
export IK2_URL=${IK2_URL:-http://127.0.0.1:8090}
export IK2_UNIT=${IK2_UNIT:-ik2-serve.service}
export IK2_GPUS=${IK2_GPUS:-}
export IK2_DASH_HOST=${IK2_DASH_HOST:-127.0.0.1}
export IK2_DASH_PORT=${IK2_DASH_PORT:-8080}
export IK2_ENERGY_SCRIPT=${IK2_ENERGY_SCRIPT:-}
exec python3 server.py
