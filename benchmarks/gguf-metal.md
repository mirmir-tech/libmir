# GGUF on Metal

mirmir serving llama.cpp's GGUF files directly on Apple Silicon, against
llama.cpp's own `llama-server` on the same files.

Measured on 2026-11-23 with the 0.7 candidate (libmir `891a79b`). llama.cpp's
cells are its 2026-11-17 run of the same files and settings; llama.cpp is
unchanged since.

## Matrix

| Property | Value |
|:---|:---|
| Machine | Apple M5 Pro, 64 GiB, macOS 27.2.1 |
| Harness | `llama-benchy` `0.4.1.dev1+ge9be34457` |
| PP | 2,048 |
| TG | 128 |
| Concurrency | 1 / 2 / 10 (Qwen3-8B Q4_K_M: 1 / 2 / 5 / 10) |
| Depth | 0 / 4,096 (Qwen3-8B Q4_K_M: 0 / 4,096 / 8,192 / 16,384 / 32,768) |
| Cache | Prefix |
| Latency | API |
| Runs | 1 warm-up + 3 measured |
| mirmir | 0.7 candidate, libmir `891a79b`, mirtal `0531428`, MLX 0.32.4 |
| Reference | llama.cpp `b1b770d90` `llama-server -ngl 99 -fa on --kv-unified -ctk bf16 -ctv bf16`, one slot per concurrent request, sharing a unified K/V pool of the capacity mirmir resolved for the model |
| K/V cache | BF16 on both engines |
| Aggregation | Geometric mean over cells both engines completed |

| Model | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|:---|---:|---:|---:|
| [Qwen3-30B-A3B Q4_K_M](#qwen3-30b-a3b-q4_k_m) | 1,784.0 / 1,196.2<br>149.1% · 6/6 wins | 94.83 / 71.09<br>133.4% · 5/6 wins | 3,135 / 4,283<br>0.732× · 6/6 wins |
| [Qwen3-30B-A3B Q8_0](#qwen3-30b-a3b-q8_0) | 1,647.7 / 1,104.5<br>149.2% · 6/6 wins | 78.83 / 53.47<br>147.4% · 6/6 wins | 3,349 / 4,639<br>0.722× · 6/6 wins |
| [GPT-OSS-20B MXFP4](#gpt-oss-20b-mxfp4) | 2,458.7 / 1,590.8<br>154.6% · 6/6 wins | 112.69 / 96.28<br>117.0% · 2/6 wins | 2,331 / 3,433<br>0.679× · 6/6 wins |
| [Gemma 3 12B IT Q4_K_M](#gemma-3-12b-it-q4_k_m) | 868.9 / 640.4<br>135.7% · 6/6 wins | 48.89 / 33.27<br>146.9% · 5/6 wins | 6,338 / 7,544<br>0.840× · 5/6 wins |
| [Qwen3-8B Q4_K_M](#qwen3-8b-q4_k_m) | 926.4 / 639.1<br>145.0% · 17/17 wins | 60.92 / 42.27<br>144.1% · 15/17 wins | 5,945 / 7,993<br>0.744× · 17/17 wins |
| [Qwen3-8B Q8_0](#qwen3-8b-q8_0) | 1,208.8 / 1,058.3<br>114.2% · 5/6 wins | 58.04 / 43.94<br>132.1% · 5/6 wins | 4,641 / 4,884<br>0.950× · 3/6 wins |
| [Llama 3.1 8B Instruct Q4_K_M](#llama-31-8b-instruct-q4_k_m) | 1,369.4 / 1,038.6<br>131.9% · 6/6 wins | 79.78 / 53.70<br>148.6% · 4/6 wins | 4,097 / 4,601<br>0.890× · 4/6 wins |
| [Mistral 7B Instruct v0.3 Q4_K_M](#mistral-7b-instruct-v03-q4_k_m) | 1,397.2 / 1,047.9<br>133.3% · 6/6 wins | 82.79 / 56.41<br>146.8% · 4/6 wins | 3,991 / 4,545<br>0.878× · 4/6 wins |

## Quality

Perplexity over the wikitext-2 test set with llama.cpp's protocol
(`llama-perplexity`: 40 chunks of 512 tokens, scoring each chunk's second
half), on the same GGUF files; Llama 4 Scout scores one 16,384-token chunk.
Lower is better; the two engines should agree to within numerical noise.

| Model | mirmir | llama.cpp | Difference |
|:---|---:|---:|---:|
| Qwen3-8B Q4_K_M | 9.3373 | 9.3337 | +0.04% |
| Qwen3-8B Q2_K | 10.2599 | 10.2460 | +0.14% |
| Qwen3-8B UD-IQ2_M | 10.3405 | 10.3253 | +0.15% |
| Qwen3-8B GSQ Q2_K | 9.9076 | 9.9086 | −0.01% |
| Qwen3-8B GSQ Q3_K_M | 9.3678 | 9.3626 | +0.06% |
| Qwen3-4B GSQ Q2_K | 14.6766 | 14.6545 | +0.15% |
| Qwen3-4B GSQ Q3_K_M | 13.3448 | 13.3352 | +0.07% |
| Qwen3-30B-A3B Q4_K_M | 8.2927 | 8.2981 | −0.07% |
| Llama 3.1 8B Instruct Q4_K_M | 7.3039 | 7.2984 | +0.08% |
| Mistral 7B Instruct v0.3 Q4_K_M | 5.8202 | 5.8378 | −0.30% |
| Mixtral 8x7B Q4_K_M | 4.0979 | 4.1003 | −0.06% |
| Gemma 3 12B IT Q4_K_M | 8.5756 | 8.5568 | +0.22% |
| Gemma 3 27B IT Q4_K_M | 7.3353 | 7.3297 | +0.08% |
| Llama 4 Scout Q2_K, 16,384-token context | 5.4990 | 5.5554 | −1.02% |

Where llama.cpp and MLX-LM disagree, mirmir follows MLX-LM:

- Phi-3.5 uses LongRoPE's long factors at every length, as MLX-LM's
  `SuScaledRoPE` does. llama.cpp switches to the short factors within 4,096
  tokens. At an 8,192-token context, where both use the long factors,
  Phi-3.5-mini Q4_K_M measures 5.3297 against llama.cpp's 5.3126.
- On Phi-3.5-MoE's MLX 4-bit checkpoint, mirmir measures 2.0710 and MLX-LM
  0.32.2 2.0814 (4 × 1,024 tokens). On the Q4_K_M GGUF, mirmir measures
  1.9115 and llama.cpp 3.2798.

GPT-OSS-20B (about 210 on both engines) and Gemma 4 instruct checkpoints
score far above their chat-format quality on raw wikitext; both engines
agree, and those values are not a quality comparison.

## Memory

Llama 4's chunked layers keep only one chunk of K/V: Scout's per-token K/V
on Metal falls from 196,608 to 49,152 bytes, and a 131,072-token session
from about 25 GB to 7.5 GB.

## Qwen3-30B-A3B Q4_K_M

GGUF file `Qwen3-30B-A3B-Q4_K_M.gguf`; 6 of 6 cells completed on both engines.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1,854.5 / 1,456.9 | 79.80 / 81.38 | 1,123 / 1,442 |
| 0 | 2 | 2,382.3 / 1,420.8 | 106.29 / 103.49 | 1,726 / 2,884 |
| 0 | 10 | 2,404.9 / 1,478.6 | 137.28 / 71.41 | 8,552 / 10,668 |
| 4,096 | 1 | 1,240.5 / 1,074.4 | 68.49 / 62.77 | 1,668 / 1,942 |
| 4,096 | 2 | 1,590.9 / 1,011.6 | 87.04 / 78.20 | 2,577 / 4,039 |
| 4,096 | 10 | 1,537.7 / 880.7 | 104.78 / 43.73 | 13,323 / 17,739 |

## Qwen3-30B-A3B Q8_0

GGUF file `Qwen3-30B-A3B-Q8_0.gguf`; 6 of 6 cells completed on both engines.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1,778.3 / 1,267.1 | 61.92 / 55.46 | 1,174 / 1,666 |
| 0 | 2 | 2,310.2 / 1,312.2 | 88.53 / 68.57 | 1,780 / 3,119 |
| 0 | 10 | 2,139.7 / 1,401.0 | 138.40 / 56.92 | 9,830 / 11,213 |
| 4,096 | 1 | 1,178.0 / 943.8 | 54.81 / 47.69 | 1,760 / 2,220 |
| 4,096 | 2 | 1,611.8 / 942.2 | 74.79 / 58.01 | 2,543 / 4,329 |
| 4,096 | 10 | 1,199.1 / 876.4 | 77.17 / 39.00 | 15,341 / 17,807 |

## GPT-OSS-20B MXFP4

GGUF file `gpt-oss-20b-MXFP4.gguf`; 6 of 6 cells completed on both engines.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 2,658.4 / 1,815.7 | 85.05 / 87.68 | 831 / 1,202 |
| 0 | 2 | 2,692.8 / 1,777.9 | 120.36 / 122.02 | 1,571 / 2,371 |
| 0 | 10 | 2,753.0 / 1,666.8 | 169.24 / 101.07 | 7,684 / 11,180 |
| 4,096 | 1 | 2,321.0 / 1,592.9 | 80.88 / 81.87 | 920 / 1,324 |
| 4,096 | 2 | 2,195.1 / 1,452.1 | 107.21 / 111.15 | 1,866 / 2,814 |
| 4,096 | 10 | 2,200.3 / 1,302.1 | 136.34 / 80.98 | 9,309 / 13,782 |

## Gemma 3 12B IT Q4_K_M

GGUF file `gemma-3-12b-it-Q4_K_M.gguf`; 6 of 6 cells completed on both engines.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1,016.3 / 692.4 | 30.17 / 29.93 | 2,025 / 2,971 |
| 0 | 2 | 932.8 / 743.7 | 53.78 / 43.21 | 4,413 / 5,497 |
| 0 | 10 | 864.2 / 751.5 | 97.30 / 38.89 | 23,801 / 19,532 |
| 4,096 | 1 | 836.2 / 702.9 | 29.24 / 29.74 | 2,454 / 2,914 |
| 4,096 | 2 | 789.7 / 653.5 | 49.12 / 41.60 | 5,187 / 6,008 |
| 4,096 | 10 | 795.3 / 387.9 | 60.25 / 21.79 | 23,949 / 33,004 |

## Qwen3-8B Q4_K_M

GGUF file `Qwen3-8B-Q4_K_M.gguf`; 17 of 20 cells completed on both engines.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1,588.6 / 1,018.0 | 47.28 / 48.15 | 1,317 / 2,065 |
| 0 | 2 | 1,376.7 / 1,078.3 | 82.65 / 70.50 | 2,993 / 3,799 |
| 0 | 5 | 1,312.2 / 1,162.8 | 141.19 / 93.71 | 7,836 / 8,389 |
| 0 | 10 | 1,353.7 / 1,115.5 | 134.54 / 64.86 | 13,980 / 14,308 |
| 4,096 | 1 | 1,217.8 / 962.8 | 41.88 / 42.64 | 1,713 / 2,177 |
| 4,096 | 2 | 1,082.4 / 854.1 | 68.77 / 61.33 | 3,790 / 4,780 |
| 4,096 | 5 | 1,079.3 / 831.6 | 102.15 / 65.45 | 9,489 / 11,646 |
| 4,096 | 10 | 1,076.2 / 793.4 | 112.73 / 44.28 | 19,030 / 19,927 |
| 8,192 | 1 | 1,022.0 / 761.0 | 38.56 / 37.82 | 2,039 / 2,749 |
| 8,192 | 2 | 967.0 / 693.2 | 59.24 / 52.44 | 4,237 / 5,890 |
| 8,192 | 5 | 897.8 / 660.6 | 83.73 / 51.72 | 11,409 / 14,635 |
| 8,192 | 10 | 871.9 / 326.6 | 79.30 / 19.76 | 23,229 / 38,607 |
| 16,384 | 1 | 664.8 / 507.5 | 32.05 / 29.63 | 3,123 / 4,107 |
| 16,384 | 2 | 669.0 / 492.7 | 46.58 / 39.26 | 6,137 / 8,292 |
| 16,384 | 5 | 669.3 / 228.6 | 55.07 / 21.44 | 15,306 / 36,306 |
| 16,384 | 10 | 208.6 / failed | 18.15 / failed | 48,415 / failed |
| 32,768 | 1 | 451.3 / 322.0 | 24.29 / 21.88 | 4,587 / 6,459 |
| 32,768 | 2 | 441.1 / 306.0 | 29.38 / 25.43 | 9,296 / 13,349 |
| 32,768 | 5 | 86.3 / failed | 9.85 / failed | 71,715 / failed |
| 32,768 | 10 | 57.7 / missing | 3.96 / missing | 171,575 / missing |

## Qwen3-8B Q8_0

GGUF file `Qwen3-8B-Q8_0.gguf`; 6 of 6 cells completed on both engines.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1,198.2 / 1,334.6 | 31.03 / 32.17 | 1,749 / 1,605 |
| 0 | 2 | 1,274.8 / 1,269.6 | 56.13 / 55.94 | 3,226 / 3,222 |
| 0 | 10 | 1,347.2 / 1,159.8 | 140.15 / 66.18 | 15,263 / 13,750 |
| 4,096 | 1 | 1,237.5 / 997.1 | 29.44 / 28.34 | 1,693 / 2,125 |
| 4,096 | 2 | 1,150.8 / 883.1 | 52.24 / 47.77 | 3,563 / 4,618 |
| 4,096 | 10 | 1,064.7 / 811.9 | 101.83 / 44.64 | 19,238 / 19,457 |

## Llama 3.1 8B Instruct Q4_K_M

GGUF file `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`; 6 of 6 cells completed on both engines.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1,638.5 / 1,262.0 | 48.58 / 50.62 | 1,272 / 1,652 |
| 0 | 2 | 1,630.9 / 1,241.5 | 86.68 / 79.34 | 2,556 / 3,302 |
| 0 | 10 | 1,430.8 / 1,122.4 | 169.32 / 55.11 | 14,562 / 11,593 |
| 4,096 | 1 | 1,285.3 / 983.2 | 43.68 / 44.11 | 1,596 / 2,085 |
| 4,096 | 2 | 1,200.8 / 885.8 | 72.12 / 62.10 | 3,414 / 4,584 |
| 4,096 | 10 | 1,117.4 / 819.6 | 114.83 / 39.56 | 18,335 / 15,710 |

## Mistral 7B Instruct v0.3 Q4_K_M

GGUF file `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`; 6 of 6 cells completed on both engines.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1,710.1 / 1,284.2 | 51.45 / 53.88 | 1,201 / 1,599 |
| 0 | 2 | 1,651.1 / 1,255.4 | 90.25 / 85.63 | 2,486 / 3,247 |
| 0 | 10 | 1,430.8 / 1,123.5 | 174.69 / 56.68 | 14,349 / 11,534 |
| 4,096 | 1 | 1,389.2 / 994.8 | 45.91 / 46.65 | 1,478 / 2,060 |
| 4,096 | 2 | 1,212.1 / 900.3 | 74.29 / 66.25 | 3,382 / 4,528 |
| 4,096 | 10 | 1,093.6 / 816.3 | 116.35 / 39.87 | 18,875 / 15,776 |

`failed`: llama-benchy exited with an error against llama-server; `missing`:
the cell did not run after that failure. Qwen3-8B Q4_K_M's 16,384- and
32,768-token cells at concurrency 5 and 10 are where mirmir is weakest; they
are left out of the ratios where llama.cpp has no value.
