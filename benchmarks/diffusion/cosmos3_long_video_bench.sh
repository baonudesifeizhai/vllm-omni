#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:-baseline}"
ENDPOINT="${ENDPOINT:-http://localhost:8092/v1/videos/sync}"
OUT_DIR="${OUT_DIR:-/tmp/cosmos3_long_bench}"

MODEL="${MODEL:-nvidia/Cosmos3-Super}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS="${FPS:-24}"
NUM_FRAMES="${NUM_FRAMES:-381}"
NUM_STEPS="${NUM_STEPS:-35}"
RUNS="${RUNS:-10}"
WARMUPS="${WARMUPS:-1}"
SEED_BASE="${SEED_BASE:-2026070300}"
KEEP_VIDEOS="${KEEP_VIDEOS:-0}"
PROMPT="${PROMPT:-A cinematic video of a red sports car driving on a mountain road at sunrise, realistic lighting, smooth camera movement}"

mkdir -p "$OUT_DIR"

OUT="$OUT_DIR/${LABEL}_video_long_${NUM_FRAMES}f.jsonl"
CONFIG="$OUT_DIR/${LABEL}_video_long_${NUM_FRAMES}f.config"
: > "$OUT"

cat > "$CONFIG" <<EOF
label=$LABEL
endpoint=$ENDPOINT
model=$MODEL
width=$WIDTH
height=$HEIGHT
fps=$FPS
num_frames=$NUM_FRAMES
num_steps=$NUM_STEPS
runs=$RUNS
warmups=$WARMUPS
seed_base=$SEED_BASE
keep_videos=$KEEP_VIDEOS
prompt=$PROMPT
EOF

run_one() {
  local kind="$1"
  local idx="$2"
  local seed="$3"
  local stem="${LABEL}_${kind}_${NUM_FRAMES}f_${idx}"
  local resp="$OUT_DIR/${stem}.mp4"
  local head="$OUT_DIR/${stem}.headers"
  local meta code latency size infer

  echo "===== $LABEL $kind $idx seed=$seed frames=$NUM_FRAMES ====="

  meta=$(curl -sS \
    -D "$head" \
    -o "$resp" \
    -w "%{http_code} %{time_total} %{size_download}" \
    -X POST "$ENDPOINT" \
    -F model="$MODEL" \
    -F prompt="$PROMPT" \
    -F width="$WIDTH" \
    -F height="$HEIGHT" \
    -F fps="$FPS" \
    -F num_frames="$NUM_FRAMES" \
    -F num_inference_steps="$NUM_STEPS" \
    -F seed="$seed")

  read -r code latency size <<< "$meta"
  infer=$(tr -d '\r' < "$head" | awk -F': ' 'tolower($1)=="x-inference-time-s"{print $2}')

  printf '{"label":"%s","kind":"%s","index":%s,"seed":%s,"http_code":%s,"latency_s":%s,"x_inference_time_s":"%s","size_download":%s,"num_frames":%s,"fps":%s,"width":%s,"height":%s,"num_steps":%s}\n' \
    "$LABEL" "$kind" "$idx" "$seed" "$code" "$latency" "$infer" "$size" "$NUM_FRAMES" "$FPS" "$WIDTH" "$HEIGHT" "$NUM_STEPS" | tee -a "$OUT"

  if [ "$code" != "200" ]; then
    echo "request failed headers:"
    cat "$head"
    echo "response body:"
    sed -n '1,120p' "$resp"
    exit 1
  fi

  if [ "$KEEP_VIDEOS" != "1" ]; then
    rm -f "$resp"
  fi
}

for i in $(seq 1 "$WARMUPS"); do
  run_one warmup "$i" "$((SEED_BASE - i))"
done

for i in $(seq 1 "$RUNS"); do
  run_one bench "$i" "$((SEED_BASE + i))"
done

python3 - "$OUT" <<'PY'
import json
import statistics
import sys

path = sys.argv[1]
xs = []
infer_xs = []
sizes = []

with open(path) as f:
    for line in f:
        row = json.loads(line)
        if row["kind"] != "bench" or row["http_code"] != 200:
            continue
        xs.append(float(row["latency_s"]))
        sizes.append(int(row["size_download"]))
        if row["x_inference_time_s"]:
            infer_xs.append(float(row["x_inference_time_s"]))

def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    idx = round((len(values) - 1) * q)
    return values[idx]

print("===== summary", path, "=====")
print("count", len(xs))
if xs:
    print("latency_mean", round(statistics.mean(xs), 6))
    print("latency_p50", round(statistics.median(xs), 6))
    print("latency_p90", round(percentile(xs, 0.90), 6))
    print("latency_min", round(min(xs), 6), "latency_max", round(max(xs), 6))
if infer_xs:
    print("x_inference_mean", round(statistics.mean(infer_xs), 6))
    print("x_inference_p50", round(statistics.median(infer_xs), 6))
if sizes:
    print("download_mb_mean", round(statistics.mean(sizes) / 1_000_000, 3))
PY
