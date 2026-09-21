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
| Driver | 580.173.02 |
| mirmir | 0.3.1 (2026-09-21) |
| Reference | vLLM 0.29.0 |
| K/V cache | BF16 on both engines |
| Token budget | mirmir 1,024 / vLLM 2,096 |
| mirmir cells | 36 |
| Reference cells | 36 |

| Metric | mirmir | vLLM | Ratio | Cell wins |
|:---|---:|---:|---:|---:|
| PP tok/s | 4,681.9 | 4,562.6 | 102.6% | 19/36 |
| TG tok/s | 75.60 | 79.25 | 95.4% | 7/36 |
| TTFT ms | 2,159 | 2,305 | 0.937× | 22/36 |

| Depth | Phase | PP mirmir | PP vLLM | PP % | TG mirmir | TG vLLM | TG % | TTFT mirmir ms | TTFT vLLM ms | TTFT × |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain | 5,568.5 | 6,172.1 | 90.2% | 114.64 | 129.53 | 88.5% | 822 | 849 | 0.968× |
| 4,096 | load | 5,514.2 | 5,842.0 | 94.4% | 95.29 | 102.28 | 93.2% | 1,737 | 1,659 | 1.047× |
| 4,096 | reuse | 4,839.8 | 3,830.4 | 126.4% | 109.50 | 109.76 | 99.8% | 982 | 1,294 | 0.759× |
| 8,192 | load | 5,489.2 | 5,524.7 | 99.4% | 70.26 | 73.22 | 96.0% | 3,397 | 3,374 | 1.007× |
| 8,192 | reuse | 4,476.2 | 3,704.1 | 120.8% | 103.99 | 104.95 | 99.1% | 1,064 | 1,338 | 0.795× |
| 16,384 | load | 5,189.8 | 5,098.9 | 101.8% | 46.25 | 47.14 | 98.1% | 7,052 | 7,124 | 0.990× |
| 16,384 | reuse | 3,930.0 | 3,496.8 | 112.4% | 94.18 | 96.98 | 97.1% | 1,212 | 1,422 | 0.853× |
| 32,768 | load | 4,582.9 | 4,537.4 | 101.0% | 26.90 | 27.29 | 98.5% | 15,724 | 15,876 | 0.990× |
| 32,768 | reuse | 3,167.2 | 3,747.5 | 84.5% | 78.81 | 88.46 | 89.1% | 1,497 | 1,387 | 1.079× |

Both engines serve ten sequences with prefix caching enabled; every one of the
486 measured responses per engine contains exactly 128 tokens. mirmir trails
most at depth 0 and in 32,768-token prefix reuse, and leads in 4,096- to
16,384-token reuse, where its 16-token checkpoint granularity recovers more of
the cached prefix than the 1,056-token blocks of the reference.

## Metal single-request diagnostic — Apple M3 Max

This focused comparison uses the corresponding
`Qwen3.5-35B-A3B-MLX-MXFP4` checkpoint, device-token greedy sampling, 64 decode
tokens, and isolated prompts without prefix retention. Mirmir reports sample
medians; MLX-LM 0.31.3 reports trial averages. It is a four-context kernel and
scheduler diagnostic, not the 36-cell API matrix above.

| Context | PP mirmir | PP MLX-LM | PP % | TG mirmir | TG MLX-LM | TG % |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 613.27 | 632.97 | 96.9% | 113.46 | 104.72 | 108.3% |
| 512 | 1,160.76 | 1,219.34 | 95.2% | 112.18 | 104.22 | 107.6% |
| 2,048 | 1,525.71 | 1,379.12 | 110.6% | 109.33 | 101.30 | 107.9% |
| 8,192 | 1,276.45 | 1,227.90 | 104.0% | 102.15 | 91.20 | 112.0% |
| Geometric mean | 1,085.09 | 1,069.22 | 101.5% | 109.19 | 100.21 | 109.0% |

The device-token pipeline evaluates logits, recurrent state and paged K/V
arenas as one explicit root generation before the next step. Three distinct
128-token prompts preserve the full 128-token reference sequence. Native U32
MXFP4 matmul/gather and compiled MXFP4 Gated Delta decode remain enabled.
The complete-step tuner selects between separate and fused shared-expert
gate/up plans from alternating measurements of the whole device pipeline. It
rejects the locally faster fused operator in every observed context bucket.
Generation leads MLX-LM in all four contexts; aggregate prefill also leads,
while the 128- and 512-token prefill cells remain below the reference.
