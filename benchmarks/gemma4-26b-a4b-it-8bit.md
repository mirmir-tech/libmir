# Gemma 4 26B-A4B IT 8-bit

## Model

| Property | Value |
|:---|:---|
| Hugging Face | `mlx-community/gemma-4-26b-a4b-it-8bit` |
| Snapshot | `33c6d23798a0af159529890f79329206dbfbd73c` |
| Parameters | 26B total / 4B active |
| Architecture | Dense plus routed-expert decoder-only Transformer |
| Checkpoint | MLX affine 8-bit, group size 64 |
| Weight size | 26.03 GiB |
| Layers | 30 |
| Attention | 25 sliding-window / 5 full-attention layers |
| Sliding attention | 16 Q heads / 8 KV heads / 256 head dimension / 1,024 window |
| Full attention | 16 Q heads / 2 KV heads / 512 head dimension |
| Hidden size | 2,816 |
| Dense intermediate size | 2,112 |
| Routed experts | 128 |
| Active experts | 8 |
| Expert intermediate size | 704 |
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
| mirmir runs | 2 warm-ups + 3 measured |
| MLX-LM runs | 1 warm-up + 3 measured |

## Metal — Apple M3 Max, 40-core GPU, 64 GiB

| Property | Value |
|:---|:---|
| Machine | MacBook Pro `Mac15,8` |
| GPU | Apple M3 Max, 40 cores |
| Memory | 64 GiB unified |
| mirmir | 0.3.1 |
| Workmir revision | `27d5621a0a49f07229baeca68a3d598adb41c88a` |
| libmir base revision | `51d42e30d1c92528531fb36d4c0f50e10d2fd61c` plus benchmarked Metal prefill changes |
| Reference | MLX-LM 0.31.3, revision `ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd` |
| mirmir cells | 36 complete |
| Reference cells | 26 complete, then process abort |

## Complete mirmir result

All 486 measured requests returned exactly 128 tokens without a request or
backend error. The geometric means across all 36 cells are 439.4 PP tok/s,
25.07 TG tok/s, and 29,113 ms TTFT.

| Depth | Phase | Cells | PP tok/s | TG tok/s | TTFT ms |
|---:|:---|---:|---:|---:|---:|
| 0 | plain | 4 | 776.4 | 64.91 | 9,160 |
| 4,096 | load | 4 | 652.2 | 47.42 | 19,884 |
| 4,096 | reuse | 4 | 596.9 | 46.22 | 10,859 |
| 8,192 | load | 4 | 581.3 | 38.37 | 44,557 |
| 8,192 | reuse | 4 | 428.1 | 31.47 | 12,808 |
| 16,384 | load | 4 | 529.3 | 19.36 | 90,121 |
| 16,384 | reuse | 4 | 301.7 | 19.85 | 18,529 |
| 32,768 | load | 4 | 461.8 | 8.31 | 185,676 |
| 32,768 | reuse | 4 | 110.1 | 7.14 | 42,926 |

The sustained tail remains a performance defect. Generation drops sharply at
16k/C10 and 32k/C5/C10, consistent with long-lived thermal or cache-residency
degradation rather than the short, cooled kernel gates.

## Partial comparison with MLX-LM

MLX-LM completed 318 requests and then terminated with `SIGABRT` on the Metal
completion queue. It did not start 16,384/C10 or any 32,768 cell. One completed
8,192/C10 reuse request generated 105 rather than 128 tokens; the other 317
generated 128. The metrics below are therefore a diagnostic comparison over
26 common positive cells, not a canonical reference result.

| Metric | mirmir | MLX-LM | Ratio | Cell wins |
|:---|---:|---:|---:|---:|
| PP tok/s | 569.0 | 362.2 | 157.1% | 25/26 |
| TG tok/s | 40.04 | 35.14 | 113.9% | 21/26 |
| TTFT ms | 18,017 | 28,815 | 0.625× | 25/26 |

Rows use all common concurrency values. Depth 16,384 contains C1/C2/C5 only.

| Depth | Phase | Cells | PP mirmir | PP MLX | PP % | TG mirmir | TG MLX | TG % | TTFT mirmir ms | TTFT MLX ms | TTFT × |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | plain | 4 | 776.4 | 731.4 | 106.2% | 64.91 | 63.17 | 102.8% | 9,160 | 9,188 | 0.997× |
| 4,096 | load | 4 | 652.2 | 509.6 | 128.0% | 47.42 | 40.95 | 115.8% | 19,884 | 25,477 | 0.780× |
| 4,096 | reuse | 4 | 596.9 | 283.1 | 210.9% | 46.22 | 40.47 | 114.2% | 10,859 | 22,908 | 0.474× |
| 8,192 | load | 4 | 581.3 | 402.5 | 144.4% | 38.37 | 30.15 | 127.3% | 44,557 | 64,515 | 0.691× |
| 8,192 | reuse | 4 | 428.1 | 182.4 | 234.7% | 31.47 | 30.17 | 104.3% | 12,808 | 35,656 | 0.359× |
| 16,384 | load | 3 | 540.4 | 462.5 | 116.9% | 27.60 | 27.97 | 98.7% | 64,716 | 76,848 | 0.842× |
| 16,384 | reuse | 3 | 439.9 | 212.2 | 207.3% | 29.34 | 20.51 | 143.0% | 10,035 | 20,398 | 0.492× |

The paired aggregate favors mirmir, especially for prefix reuse, but both
long-lived runs show high-concurrency degradation and the MLX-LM process
crashed. A complete MLX-LM rerun with stderr capture and unified-memory
telemetry is required before this model can enter the canonical headline
comparison.

The aggregate does not mean single-request kernel parity. At depth 0/C1,
mirmir reaches 937.2 versus 1,107.9 PP tok/s, 51.31 versus 53.21 TG tok/s, and
2,188 versus 1,851 ms TTFT. Its paired advantage comes from sustained
concurrency and prefix reuse, while both engines degrade in the largest cells.

The MLX-LM result was reconstructed from its progress event stream. Repeating
the reconstruction on the complete mirmir stream reproduces the native result
within 0.0024% PP, 0.080% TG, and effectively exact TTFT.
