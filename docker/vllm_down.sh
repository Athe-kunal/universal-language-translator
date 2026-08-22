#!/usr/bin/env bash
# Tears down a vllm server container started by vllm_up.sh.
# Invoke via `make vllm-down ...` (see Makefile) rather than running directly.
#
# Identify the container by NAME (as printed by vllm_up.sh / `make vllm-up`), or by
# PORT (the host port it was published on) if you don't have the name handy. At least
# one of the two is required.
#
# Usage:
#   NAME=vllm-gpu0-1 vllm_down.sh
#   PORT=18000 vllm_down.sh
set -euo pipefail

NAME="${NAME:-}"
PORT="${PORT:-}"

if [ -z "$NAME" ] && [ -z "$PORT" ]; then
  echo "Set NAME=<container> or PORT=<host port> to identify the container to tear down." >&2
  exit 1
fi

if [ -z "$NAME" ]; then
  NAME="$(docker ps -a --filter "label=vllm-server" --filter "publish=$PORT" --format '{{.Names}}' | head -n1)"
  if [ -z "$NAME" ]; then
    echo "No vllm-server container found publishing port $PORT." >&2
    exit 1
  fi
fi

if ! docker inspect "$NAME" >/dev/null 2>&1; then
  echo "No container named $NAME found." >&2
  exit 1
fi

echo "Stopping and removing $NAME ..." >&2
docker rm -f "$NAME" >/dev/null
echo "Done." >&2
