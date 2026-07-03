#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_SCRIPT="$ROOT_DIR/benchmarks/diffusion/cosmos3_long_video_bench.sh"

OUT_DIR="${OUT_DIR:-/tmp/cosmos3_long_bench}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8092}"
MODEL="${MODEL:-nvidia/Cosmos3-Super}"
CUDA_DEVICES="${CUDA_DEVICES:-4,5,6,7}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-4}"
RING_DEGREE="${RING_DEGREE:-1}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-900}"
KILL_EXISTING="${KILL_EXISTING:-1}"
VLLM_OMNI_BIN="${VLLM_OMNI_BIN:-vllm-omni}"

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

SERVER_PID=""

mkdir -p "$OUT_DIR"

kill_port_server() {
  local pids
  pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
    if [ -n "$pids" ]; then
      echo "killing existing server on port $PORT: $pids"
      kill $pids 2>/dev/null || true
      sleep 5
      pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
      if [ -n "$pids" ]; then
        kill -9 $pids 2>/dev/null || true
      fi
    fi
  elif command -v fuser >/dev/null 2>&1; then
    echo "killing existing server on port $PORT with fuser"
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    sleep 5
  else
    echo "lsof/fuser not found; falling back to model-name pkill"
    pkill -f "$VLLM_OMNI_BIN serve $MODEL" 2>/dev/null || true
    sleep 5
  fi
}

stop_server() {
  if [ -n "${SERVER_PID:-}" ]; then
    echo "stopping server pid/group $SERVER_PID"
    kill -- "-$SERVER_PID" 2>/dev/null || kill "$SERVER_PID" 2>/dev/null || true
    sleep 8
    kill -9 -- "-$SERVER_PID" 2>/dev/null || kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
  kill_port_server
}

cleanup() {
  stop_server
}
trap cleanup EXIT

wait_for_server() {
  local label="$1"
  local log="$2"
  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  local next_log=$SECONDS

  echo "waiting for $label server: http://127.0.0.1:$PORT/health"
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "$label server is ready"
      return 0
    fi

    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "$label server exited while starting. Last log lines:"
      tail -n 120 "$log" || true
      exit 1
    fi

    if [ "$SECONDS" -ge "$next_log" ]; then
      echo "still waiting for $label server..."
      tail -n 20 "$log" || true
      next_log=$((SECONDS + 30))
    fi
    sleep 5
  done

  echo "$label server did not become ready in ${STARTUP_TIMEOUT_S}s. Last log lines:"
  tail -n 160 "$log" || true
  exit 1
}

start_server() {
  local label="$1"
  local use_symmem="$2"
  local log="$OUT_DIR/${label}_${NUM_FRAMES}f_server.log"

  if [ "$KILL_EXISTING" = "1" ]; then
    kill_port_server
  fi

  echo "===== starting $label server ====="
  echo "log: $log"

  if [ "$use_symmem" = "1" ]; then
    setsid env \
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
      VLLM_OMNI_USE_SYMM_MEM_ALL2ALL=1 \
      VLLM_OMNI_USE_SYMM_MEM_PACKED_QKV_ALL2ALL=1 \
      VLLM_OMNI_USE_SYMM_MEM_ASYNC_ULYSSES=1 \
      VLLM_OMNI_USE_SYMM_MEM_ATTENTION_OVERLAP=1 \
      VLLM_OMNI_SYMM_MEM_ATTENTION_CHUNKS="${VLLM_OMNI_SYMM_MEM_ATTENTION_CHUNKS:-2}" \
      "$VLLM_OMNI_BIN" serve "$MODEL" \
        --omni \
        --host "$HOST" \
        --port "$PORT" \
        --ulysses-degree "$ULYSSES_DEGREE" \
        --ring-degree "$RING_DEGREE" \
        --no-guardrails \
        > "$log" 2>&1 &
  else
    setsid env \
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
      VLLM_OMNI_USE_SYMM_MEM_ALL2ALL=0 \
      VLLM_OMNI_USE_SYMM_MEM_PACKED_QKV_ALL2ALL=0 \
      VLLM_OMNI_USE_SYMM_MEM_ASYNC_ULYSSES=0 \
      VLLM_OMNI_USE_SYMM_MEM_ATTENTION_OVERLAP=0 \
      "$VLLM_OMNI_BIN" serve "$MODEL" \
        --omni \
        --host "$HOST" \
        --port "$PORT" \
        --ulysses-degree "$ULYSSES_DEGREE" \
        --ring-degree "$RING_DEGREE" \
        --no-guardrails \
        > "$log" 2>&1 &
  fi

  SERVER_PID=$!
  wait_for_server "$label" "$log"
}

run_bench() {
  local label="$1"
  ENDPOINT="http://127.0.0.1:$PORT/v1/videos/sync" \
  OUT_DIR="$OUT_DIR" \
  MODEL="$MODEL" \
  WIDTH="$WIDTH" \
  HEIGHT="$HEIGHT" \
  FPS="$FPS" \
  NUM_FRAMES="$NUM_FRAMES" \
  NUM_STEPS="$NUM_STEPS" \
  RUNS="$RUNS" \
  WARMUPS="$WARMUPS" \
  SEED_BASE="$SEED_BASE" \
  KEEP_VIDEOS="$KEEP_VIDEOS" \
  PROMPT="$PROMPT" \
    "$BENCH_SCRIPT" "$label"
}

compare_results() {
  python3 - "$OUT_DIR" "$NUM_FRAMES" <<'PY'
import json
import statistics
import sys

out_dir, num_frames = sys.argv[1], sys.argv[2]

def load(label):
    path = f"{out_dir}/{label}_video_long_{num_frames}f.jsonl"
    values = []
    infer = []
    sizes = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["kind"] != "bench" or row["http_code"] != 200:
                continue
            values.append(float(row["latency_s"]))
            sizes.append(int(row["size_download"]))
            if row["x_inference_time_s"]:
                infer.append(float(row["x_inference_time_s"]))
    return values, infer, sizes

def p90(values):
    values = sorted(values)
    return values[round((len(values) - 1) * 0.90)]

stats = {}
for label in ("baseline", "symmem"):
    values, infer, sizes = load(label)
    stats[label] = values
    print(label)
    print("  count", len(values))
    if values:
        print("  mean", round(statistics.mean(values), 6))
        print("  p50", round(statistics.median(values), 6))
        print("  p90", round(p90(values), 6))
        print("  min", round(min(values), 6), "max", round(max(values), 6))
    if infer:
        print("  x_inference_mean", round(statistics.mean(infer), 6))
    if sizes:
        print("  download_mb_mean", round(statistics.mean(sizes) / 1_000_000, 3))

if stats["baseline"] and stats["symmem"]:
    b = statistics.mean(stats["baseline"])
    s = statistics.mean(stats["symmem"])
    print("speedup")
    print("  latency_delta_s", round(b - s, 6))
    print("  latency_reduction_pct", round((b - s) / b * 100, 3))
    print("  throughput_speedup", round(b / s, 4))
PY
}

echo "===== cosmos3 long video benchmark config ====="
cat <<EOF
out_dir=$OUT_DIR
model=$MODEL
cuda_devices=$CUDA_DEVICES
port=$PORT
ulysses_degree=$ULYSSES_DEGREE
ring_degree=$RING_DEGREE
width=$WIDTH
height=$HEIGHT
fps=$FPS
num_frames=$NUM_FRAMES
num_steps=$NUM_STEPS
runs=$RUNS
warmups=$WARMUPS
seed_base=$SEED_BASE
keep_videos=$KEEP_VIDEOS
EOF

start_server baseline 0
run_bench baseline
stop_server

start_server symmem 1
run_bench symmem
stop_server

compare_results | tee "$OUT_DIR/compare_${NUM_FRAMES}f.txt"
