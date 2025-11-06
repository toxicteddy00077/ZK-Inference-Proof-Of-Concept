#!/bin/bash
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
images_dir="$project_root/images"
logs_dir="$project_root/logs/client_logs"
SERVER="${1:-http://127.0.0.1:50051}"
if [ -n "$2" ]; then
    MAX_CONCURRENT="$2"
else
    nproc_val=$(nproc 2>/dev/null || echo 2)
    MAX_CONCURRENT=$(( nproc_val / 2 ))
    [ "$MAX_CONCURRENT" -lt 1 ] && MAX_CONCURRENT=1
fi
mkdir -p "$logs_dir"
if compgen -G "$logs_dir"/*.log > /dev/null; then rm -f "$logs_dir"/*.log; fi
mapfile -t IMAGES < <(find "$images_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort)
NUM_CLIENTS=${#IMAGES[@]}
if [ "$NUM_CLIENTS" -eq 0 ]; then echo "No images found in $images_dir"; exit 1; fi
echo "Starting $NUM_CLIENTS clients (concurrency=$MAX_CONCURRENT)"
pids=()
for i in "${!IMAGES[@]}"; do
    while [ $(jobs -rp | wc -l) -ge "$MAX_CONCURRENT" ]; do sleep 0.3; done
    IMAGE="${IMAGES[$i]}"
    base="$(basename "$IMAGE")"
    timestamp=$(date +%s)
    LOG="$logs_dir/client_${i}_${timestamp}_${base}.log"
    ( cargo run --manifest-path "$project_root/client/Cargo.toml" --quiet -- "$SERVER" "$IMAGE" > "$LOG" 2>&1 ) &
    pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
echo "All clients finished. Logs: $logs_dir"
ls -1 "$logs_dir" | sed -n '1,200p'

