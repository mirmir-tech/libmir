# GPT-OSS-20B BF16

## Model

| Property | Value |
|:---|:---|
| Hugging Face | `unsloth/gpt-oss-20b-BF16` |
| Snapshot | `cc89b3e7fd423253264883a80a4fa5abc619649f` |
| Parameters | 20.9B |
| Architecture | Sparse MoE decoder-only Transformer |
| Checkpoint | BF16 SafeTensors |
| Weight size | 38.96 GiB |
| Routed experts | 32 |
| Active experts | 4 |
| Q heads | 64 |
| KV heads | 8 |
| Head dimension | 64 |
| Attention | Alternating 128-token sliding-window / full |
| Context | 131,072 |

## Matrix

| Property | Value |
|:---|:---|
| Harness | `llama-benchy` `0.4.1.dev1+ge9be34457` |
| Harness revision | `e9be344578cec17745066b220798b80a0d2686d3` |
| PP | 2,048 |
| TG | 128 |
| Concurrency | 1 / 2 / 5 / 10 |
| Depth | 0 / 4,096 / 8,192 / 16,384 / 32,768 / 65,535 / 100,000 |
| Cache | Prefix |
| Latency | API |
| Runs | 1 warm-up + 3 measured |

## CUDA — NVIDIA GX10 (GB10)

| Property | Value |
|:---|:---|
| Machine | NVIDIA GX10 (GB10) |
| Driver | 580.159.03 |
| Reference | vLLM 0.25.1 |
| mirmir cells | 52 |
| Reference cells | 52 |

| Metric | mirmir | vLLM | Ratio | Cell wins |
|:---|---:|---:|---:|---:|
| PP tok/s | 2,827.2 | 2,254.3 | 125.4% | 43/52 |
| TG tok/s | 27.74 | 25.53 | 108.7% | 26/52 |
| TTFT ms | 6,088 | 7,422 | 0.820× | 44/52 |

| Depth | Phase | PP mirmir | PP vLLM | PP % | TG mirmir | TG vLLM | TG % | TTFT mirmir ms | TTFT vLLM ms | TTFT × |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain | 3,502.7 | 3,924.3 | 89.3% | 46.51 | 47.27 | 98.4% | 1,763 | 1,578 | 1.117× |
| 4,096 | load | 3,516.2 | 3,858.9 | 91.1% | 37.97 | 41.52 | 91.5% | 3,045 | 2,923 | 1.042× |
| 4,096 | reuse | 3,240.7 | 3,016.3 | 107.4% | 45.39 | 46.26 | 98.1% | 1,914 | 2,070 | 0.925× |
| 8,192 | load | 4,098.2 | 3,701.1 | 110.7% | 34.37 | 33.88 | 101.4% | 5,219 | 5,657 | 0.923× |
| 8,192 | reuse | 3,090.1 | 2,723.6 | 113.5% | 44.62 | 45.07 | 99.0% | 2,024 | 2,299 | 0.880× |
| 16,384 | load | 4,026.1 | 3,261.5 | 123.4% | 27.25 | 22.91 | 119.0% | 10,925 | 11,886 | 0.919× |
| 16,384 | reuse | 2,754.8 | 2,247.5 | 122.6% | 42.15 | 42.44 | 99.3% | 2,310 | 2,770 | 0.834× |
| 32,768 | load | 3,671.1 | 2,658.9 | 138.1% | 18.70 | 14.01 | 133.5% | 22,912 | 28,141 | 0.814× |
| 32,768 | reuse | 2,196.5 | 1,690.7 | 129.9% | 38.80 | 38.43 | 101.0% | 2,874 | 3,672 | 0.783× |
| 65,535 | load | 3,112.1 | 1,923.7 | 161.8% | 9.98 | 7.01 | 142.4% | 50,426 | 76,623 | 0.658× |
| 65,535 | reuse | 1,619.4 | 1,107.9 | 146.2% | 32.08 | 32.22 | 99.6% | 3,900 | 5,555 | 0.702× |
| 100,000 | load | 2,631.8 | 1,481.7 | 177.6% | 6.37 | 4.19 | 152.1% | 87,476 | 151,142 | 0.579× |
| 100,000 | reuse | 1,229.5 | 810.1 | 151.8% | 27.53 | 28.47 | 96.7% | 5,090 | 7,637 | 0.666× |
