# Qwen3.6-35B-A3B NVFP4

## Model

| Property | Value |
|:---|:---|
| Hugging Face | `nvidia/Qwen3.6-35B-A3B-NVFP4` |
| Snapshot | `491c2f1ea524c639598bf8fa787a93fed5a6fbce` |
| Parameters | 35B total / 3B active |
| Architecture | Hybrid linear/full-attention sparse MoE VLM |
| Checkpoint | Mixed FP8/NVFP4 SafeTensors |
| Weight size | 21.82 GiB |
| Layers | 40 |
| Full attention | Every fourth layer |
| Hidden size | 2,048 |
| Routed experts | 256 |
| Active experts | 8 |
| MoE intermediate size | 512 |
| Shared expert intermediate size | 512 |
| Q heads | 16 |
| KV heads | 2 |
| Head dimension | 256 |
| Context | 262,144 |

## Matrix

| Property | Value |
|:---|:---|
| Harness | `llama-benchy` `0.4.1.dev1+ge9be34457` |
| Harness revision | `e9be344578cec17745066b220798b80a0d2686d3` |
| PP | 2,048 |
| TG | 128 |
| Concurrency | 1 / 2 / 5 / 10 |
| Depth | 0 / 4,096 / 8,192 / 16,384 / 32,768 |
| Cache | Prefix |
| Latency | API |
| Runs | 1 warm-up + 3 measured |

## CUDA — NVIDIA GX10 (GB10)

| Property | Value |
|:---|:---|
| Machine | NVIDIA GX10 (GB10) |
| Driver | 580.159.03 |
| mirmir | 0.3.1 |
| Reference | vLLM 0.25.1 |
| mirmir cells | 36 |
| Reference cells | 36 |

| Metric | mirmir | vLLM | Ratio | Cell wins |
|:---|---:|---:|---:|---:|
| PP tok/s | 2,939.3 | 2,399.0 | 122.5% | 21/36 |
| TG tok/s | 41.60 | 54.10 | 76.9% | 8/36 |
| TTFT ms | 3,379 | 3,899 | 0.867× | 21/36 |

| Depth | Phase | PP mirmir | PP vLLM | PP % | TG mirmir | TG vLLM | TG % | TTFT mirmir ms | TTFT vLLM ms | TTFT × |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain | 3,254.5 | 5,705.0 | 57.0% | 66.40 | 112.75 | 58.9% | 1,643 | 889 | 1.848× |
| 4,096 | load | 3,331.5 | 2,752.2 | 121.0% | 50.75 | 60.43 | 84.0% | 2,709 | 3,022 | 0.896× |
| 4,096 | reuse | 2,822.6 | 1,340.7 | 210.5% | 56.65 | 60.17 | 94.1% | 1,597 | 3,101 | 0.515× |
| 8,192 | load | 3,492.6 | 3,718.2 | 93.9% | 39.57 | 52.01 | 76.1% | 5,170 | 4,584 | 1.128× |
| 8,192 | reuse | 2,704.4 | 1,315.7 | 205.5% | 53.94 | 59.06 | 91.3% | 1,666 | 3,164 | 0.527× |
| 16,384 | load | 3,455.3 | 4,356.1 | 79.3% | 27.73 | 39.86 | 69.6% | 10,445 | 8,029 | 1.301× |
| 16,384 | reuse | 2,473.9 | 1,218.4 | 203.0% | 49.24 | 56.59 | 87.0% | 1,823 | 3,447 | 0.529× |
| 32,768 | load | 3,190.1 | 4,393.7 | 72.6% | 16.89 | 26.41 | 64.0% | 22,644 | 16,244 | 1.394× |
| 32,768 | reuse | 2,077.5 | 1,096.1 | 189.5% | 40.01 | 53.05 | 75.4% | 2,177 | 3,837 | 0.567× |
