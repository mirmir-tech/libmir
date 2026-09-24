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
| Driver | 580.173.02 |
| mirmir | 0.3.1 (2026-09-24) |
| Reference | vLLM 0.29.0 |
| K/V cache | BF16 on both engines; mirmir blocks of 16 tokens sized from the memory estimate (the sink-attention runtime keeps its pages) |
| Token budget | mirmir 1,024 / vLLM 2,096 |
| mirmir cells | 52 |
| Reference cells | 52 |

| Metric | mirmir | vLLM | Ratio | Cell wins |
|:---|---:|---:|---:|---:|
| PP tok/s | 2,315.1 | 1,159.8 | 199.6% | 34/52 |
| TG tok/s | 36.64 | 19.49 | 188.0% | 49/52 |
| TTFT ms | 8,285 | 13,525 | 0.613× | 27/52 |

| Depth | Phase | PP mirmir | PP vLLM | PP % | TG mirmir | TG vLLM | TG % | TTFT mirmir ms | TTFT vLLM ms | TTFT × |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain | 3,112.5 | 3,992.2 | 78.0% | 52.02 | 51.29 | 101.4% | 2,169 | 1,560 | 1.390× |
| 4,096 | load | 3,064.1 | 3,948.7 | 77.6% | 51.14 | 44.19 | 115.7% | 4,314 | 2,878 | 1.499× |
| 4,096 | reuse | 2,758.9 | 1,307.0 | 211.1% | 50.47 | 40.22 | 125.5% | 2,432 | 4,166 | 0.584× |
| 8,192 | load | 3,095.3 | 3,760.7 | 82.3% | 48.93 | 35.39 | 138.2% | 8,459 | 5,591 | 1.513× |
| 8,192 | reuse | 2,584.6 | 774.2 | 333.8% | 48.41 | 32.96 | 146.9% | 2,580 | 6,635 | 0.389× |
| 16,384 | load | 3,016.8 | 3,382.0 | 89.2% | 46.54 | 24.45 | 190.3% | 17,267 | 11,563 | 1.493× |
| 16,384 | reuse | 2,331.1 | 372.3 | 626.2% | 45.62 | 22.98 | 198.6% | 2,847 | 13,131 | 0.217× |
| 32,768 | load | 2,802.2 | 2,797.2 | 100.2% | 37.51 | 14.60 | 256.9% | 36,060 | 26,866 | 1.342× |
| 32,768 | reuse | 1,945.2 | 240.3 | 809.5% | 41.43 | 14.10 | 293.9% | 3,431 | 26,323 | 0.130× |
| 65,535 | load | 2,396.3 | 2,064.7 | 116.1% | 19.68 | 7.33 | 268.6% | 77,819 | 71,517 | 1.088× |
| 65,535 | reuse | 1,318.6 | 1,238.0 | 106.5% | 33.49 | 32.59 | 102.8% | 4,804 | 4,881 | 0.984× |
| 100,000 | load | 2,082.3 | 1,605.7 | 129.7% | 11.38 | 4.44 | 256.1% | 132,240 | 139,435 | 0.948× |
| 100,000 | reuse | 1,033.6 | 33.0 | 3136.1% | 27.27 | 4.49 | 607.7% | 5,806 | 139,701 | 0.042× |

Both engines serve ten sequences with prefix caching enabled; every one of the
702 measured responses per engine contains exactly 128 tokens. vLLM 0.29.0
runs this model on its Triton attention backend, because its FlashInfer
decode kernel for attention sinks has no build for this GPU, and it reported a
prefix cache hit rate of under 3% throughout: its reuse cells are cold
prefills, so the reuse rows and the headline overstate mirmir's lead. The
load rows are the fair comparison: mirmir is at 78–100% of vLLM's prompt
processing up to 32,768 tokens of depth and 116–130% at 65,535 and 100,000,
and leads decode on every load row (116–269%). A CUDA prefill wave now holds
every admitted row (AD-030); the table published earlier the same day had
concurrent prompts prefilling one after another, at PP 70–98% and TG 84–95%
on the load rows up to 32,768 tokens.

