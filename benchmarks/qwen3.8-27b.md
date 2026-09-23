# Qwen3.8-27B

## Model

| Property | Value |
|:---|:---|
| Hugging Face | `Qwen/Qwen3.8-27B` |
| Snapshot | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Parameters | 27B dense |
| Architecture | Hybrid linear/full-attention dense VLM |
| Checkpoint | BF16 SafeTensors, 18 shards |
| Weight size | 51.75 GiB |
| Layers | 64 |
| Full attention | Every fourth layer |
| Hidden size | 5,120 |
| Intermediate size | 17,408 |
| Q heads | 24 |
| KV heads | 4 |
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
| mirmir | 0.3.1 (2026-09-23) |
| Reference | vLLM 0.29.0 |
| K/V cache | BF16 on both engines; mirmir 24,936 blocks of 16 tokens sized from measured memory (AD-027, AD-028) |
| Token budget | mirmir 1,024 / vLLM 2,096 |
| mirmir cells | 36 |
| Reference cells | 36 |

| Metric | mirmir | vLLM | Ratio | Cell wins |
|:---|---:|---:|---:|---:|
| PP tok/s | 1,249.1 | 1,067.3 | 117.0% | 36/36 |
| TG tok/s | 9.02 | 8.99 | 100.3% | 20/36 |
| TTFT ms | 8,001 | 10,218 | 0.783× | 36/36 |

| Depth | Phase | PP mirmir | PP vLLM | PP % | TG mirmir | TG vLLM | TG % | TTFT mirmir ms | TTFT vLLM ms | TTFT × |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain | 1,407.5 | 1,281.5 | 109.8% | 11.57 | 11.96 | 96.8% | 3,236 | 4,215 | 0.768× |
| 4,096 | load | 1,399.0 | 1,228.7 | 113.9% | 10.40 | 10.50 | 99.0% | 6,821 | 8,320 | 0.820× |
| 4,096 | reuse | 1,246.7 | 1,116.7 | 111.6% | 11.35 | 11.37 | 99.8% | 3,735 | 4,570 | 0.817× |
| 8,192 | load | 1,399.1 | 1,242.1 | 112.6% | 8.62 | 8.48 | 101.7% | 13,317 | 15,361 | 0.867× |
| 8,192 | reuse | 1,196.8 | 972.7 | 123.0% | 11.11 | 11.28 | 98.5% | 3,888 | 5,514 | 0.705× |
| 16,384 | load | 1,351.6 | 1,193.9 | 113.2% | 6.50 | 6.24 | 104.3% | 27,124 | 30,889 | 0.878× |
| 16,384 | reuse | 1,117.0 | 839.7 | 133.0% | 10.68 | 10.70 | 99.8% | 4,158 | 6,332 | 0.657× |
| 32,768 | load | 1,248.3 | 1,113.3 | 112.1% | 4.41 | 4.22 | 104.6% | 57,954 | 65,520 | 0.885× |
| 32,768 | reuse | 955.7 | 758.3 | 126.0% | 9.84 | 9.99 | 98.5% | 4,812 | 6,981 | 0.689× |

Both engines serve ten sequences with prefix caching enabled; every one of the
486 measured responses per engine contains exactly 128 tokens. mirmir sizes its
K/V cache from memory measured after warm-up (AD-027) once its layer scratch,
packed decode states and decode attention workspace are shared per shape and
prefix checkpoints are bounded by a quarter of the pages (AD-028): on this
121 GiB device that leaves 399k tokens of pages, while vLLM reserves 80% of
memory for itself. mirmir leads prompt processing and first-token latency in
every cell; decode is at parity overall, ahead in every load cell from 8,192
tokens up and within 3.5% elsewhere. The host stayed above 19.6 GiB free
throughout.

