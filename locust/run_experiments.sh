#!/usr/bin/env bash
# Runs the same Locust flood against 1, 2 and 3 API containers and saves CSVs in results/.
# Usage: bash locust/run_experiments.sh [users] [spawn_rate] [duration]
set -euo pipefail
USERS=${1:-100}; RATE=${2:-10}; TIME=${3:-60s}
HOST=${HOST:-http://localhost:8000}
mkdir -p results
for N in 1 2 3; do
  echo "=== $N API container(s) ==="
  docker compose up -d --scale api=$N gateway api
  sleep 25   # let the containers load the model and nginx re-resolve DNS
  locust -f locust/locustfile.py --headless -u "$USERS" -r "$RATE" -t "$TIME" \
         --host "$HOST" --csv "results/containers_$N" --only-summary
done
python locust/summarize_results.py
