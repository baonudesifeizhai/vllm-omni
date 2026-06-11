# Wan2.2 Text-to-Video

> Text-to-video serving with Wan2.2 14B, including NVIDIA ModelOpt NVFP4

## Summary

- Vendor: Wan-AI / NVIDIA
- Model: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Quantized checkpoint: `nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4`
- Task: Text-to-video generation
- Mode: Online serving with the OpenAI-compatible video API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to serve Wan2.2 text-to-video with vLLM-Omni and
compare the BF16 checkpoint against the pre-quantized NVIDIA ModelOpt NVFP4
checkpoint. The NVFP4 checkpoint is loaded as a pre-quantized checkpoint; do
not pass `--quantization fp8`.

## References

- BF16 model card: <https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers>
- NVIDIA NVFP4 model card: <https://huggingface.co/nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4>
- Video API guide: [`docs/serving/videos_api.md`](../../docs/serving/videos_api.md)
- Online serving example:
  [`examples/online_serving/text_to_video/README.md`](../../examples/online_serving/text_to_video/README.md)

## Hardware Support

## GPU

### 1x NVIDIA Blackwell GPU

#### Environment

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA CUDA environment with Blackwell FP4 support
- vLLM version: Match the vLLM-Omni checkout used for deployment
- vLLM-Omni version or commit: Use the commit that contains ModelOpt NVFP4
  checkpoint loading for Wan2.2 diffusion transformers

#### Command

Start the BF16 server:

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni \
  --host 0.0.0.0 \
  --port 8091 \
  --trust-remote-code
```

Start the NVIDIA ModelOpt NVFP4 server:

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4 \
  --omni \
  --host 0.0.0.0 \
  --port 8091 \
  --trust-remote-code
```

If you download checkpoints locally, replace the model argument with the local
directory, for example `/root/zdj/models/Wan2.2-T2V-A14B-Diffusers-NVFP4`.

#### Verification

Run a synchronous smoke test after the server is ready:

```bash
curl -X POST http://localhost:8091/v1/videos/sync \
  -F model=nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4 \
  -F prompt="A red fox running through a snowy forest at sunrise, cinematic." \
  -F width=832 \
  -F height=480 \
  -F num_frames=33 \
  -F fps=16 \
  -F num_inference_steps=50 \
  -F guidance_scale=3.5 \
  -F guidance_scale_2=4.0 \
  -F boundary_ratio=0.875 \
  -F flow_shift=12.0 \
  -F seed=42 \
  --output wan22_t2v_nvfp4.mp4
```

Check that:

- The server responds on `http://localhost:8091/health`.
- The response is an MP4 file.
- Logs show ModelOpt NVFP4 checkpoint loading; no runtime BF16-to-FP8
  quantization flag is needed.

#### Benchmark

The numbers below are local single-GPU request-level measurements using the
serving benchmark. Each server was loaded once, then measured with 8 sequential
requests.

```text
Endpoint: /v1/videos
Task: text-to-video
Requests: 8
Concurrency: 1
Resolution: 832x480
Frames: 33
Denoising steps: 50
Warmup: 1 request, 1 denoising step
```

| Checkpoint | Mean latency | Stage generation mean | Peak VRAM |
| --- | ---: | ---: | ---: |
| BF16 | 82.087s | 81.101s | 75,762 MB |
| NVIDIA ModelOpt NVFP4 | 62.052s | 61.154s | 43,498 MB |

Example benchmark command:

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
  --endpoint /v1/videos \
  --dataset custom \
  --dataset-path /tmp/wan22_pr3305_8prompts.jsonl \
  --task t2v \
  --model nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4 \
  --port 8091 \
  --num-prompts 8 \
  --max-concurrency 1 \
  --request-rate inf \
  --warmup-requests 1 \
  --warmup-num-inference-steps 1 \
  --warmup-concurrency 1 \
  --width 832 \
  --height 480 \
  --num-frames 33 \
  --fps 16 \
  --num-inference-steps 50 \
  --seed 42 \
  --output-file /tmp/wan22_nvfp4_480x832_f33_s50_c1.json
```

#### Notes

- ModelOpt NVFP4 is a pre-quantized checkpoint format. Do not pass
  `--quantization fp8`.
- The checkpoint `quantization_config` selects the ModelOpt NVFP4 path.
- BF16, NVFP4, and any other quantized checkpoint should be compared with the
  same prompt, resolution, frame count, seed, scheduler settings, and denoising
  steps.
- For Wan2.2 480p tests, `boundary_ratio=0.875` and `flow_shift=12.0` are the
  commonly used request settings.
