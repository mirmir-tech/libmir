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
| K/V cache | BF16 on both engines; mirmir 13,425 blocks of 16 tokens sized from measured memory (AD-027, AD-028) |
| Token budget | mirmir 1,024 / vLLM 2,096 |
| mirmir cells | 36 |
| Reference cells | 36 |

| Metric | mirmir | vLLM | Ratio | Cell wins |
|:---|---:|---:|---:|---:|
| PP tok/s | 1,162.4 | 1,067.3 | 108.9% | 35/36 |
| TG tok/s | 8.65 | 8.99 | 96.2% | 18/36 |
| TTFT ms | 8,582 | 10,218 | 0.840× | 35/36 |

| Depth | Phase | PP mirmir | PP vLLM | PP % | TG mirmir | TG vLLM | TG % | TTFT mirmir ms | TTFT vLLM ms | TTFT × |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain | 1,402.9 | 1,281.5 | 109.5% | 11.58 | 11.96 | 96.8% | 3,248 | 4,215 | 0.770× |
| 4,096 | load | 1,392.9 | 1,228.7 | 113.4% | 10.40 | 10.50 | 99.0% | 6,851 | 8,320 | 0.823× |
| 4,096 | reuse | 1,243.7 | 1,116.7 | 111.4% | 11.33 | 11.37 | 99.7% | 3,746 | 4,570 | 0.820× |
| 8,192 | load | 1,394.4 | 1,242.1 | 112.3% | 8.62 | 8.48 | 101.7% | 13,363 | 15,361 | 0.870× |
| 8,192 | reuse | 1,194.8 | 972.7 | 122.8% | 11.11 | 11.28 | 98.5% | 3,894 | 5,514 | 0.706× |
| 16,384 | load | 1,347.6 | 1,193.9 | 112.9% | 6.49 | 6.24 | 104.1% | 27,210 | 30,889 | 0.881× |
| 16,384 | reuse | 1,114.9 | 839.7 | 132.8% | 10.69 | 10.70 | 99.9% | 4,165 | 6,332 | 0.658× |
| 32,768 | load | 1,236.6 | 1,113.3 | 111.1% | 4.38 | 4.22 | 104.0% | 58,526 | 65,520 | 0.893× |
| 32,768 | reuse | 515.0 | 758.3 | 67.9% | 6.81 | 9.99 | 68.1% | 8,779 | 6,981 | 1.257× |

Both engines serve ten sequences with prefix caching enabled; every one of the
486 measured responses per engine contains exactly 128 tokens. mirmir sizes its
K/V cache from memory measured after warm-up (AD-027) once its layer scratch
and packed decode states are shared per shape (AD-028): on this 121 GiB device
that leaves 215k tokens of pages and an equal prefix-checkpoint budget, while
vLLM reserves 80% of memory for itself. mirmir leads prompt processing in 35
of 36 cells and first-token latency in 35; decode is within 4% overall and
ahead in every load cell from 8,192 tokens up. The one loss is reuse at
32,768 tokens with ten requests: ten 34,842-token sessions need 348k tokens of
pages, more than the cache holds, so requests wait for pages and re-prefill
the prefix. Sharing the pages of a common prefix between sessions is the next
target.

