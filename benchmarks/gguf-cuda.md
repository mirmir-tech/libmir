# GGUF on CUDA

mirmir serving llama.cpp's GGUF files directly on NVIDIA GB10, against
llama.cpp's own `llama-server` on the same files.

> **Status.** Measured on 2026-11-23 with libmir `891a79b`. llama.cpp did not
> change between the runs this page draws on, so its cells for seven of the
> first eight models and for Mistral Small 4 come from earlier measurements of
> the same build on the same machine (2026-11-16 and 2026-11-21); Qwen3-8B
> Q4_K_M's and the other models' llama.cpp cells were measured in this run.

## Matrix

| Property | Value |
|:---|:---|
| Machine | NVIDIA GX10 (GB10), 128 GB unified memory, driver 580.173.02 |
| Harness | `llama-benchy` `0.4.1.dev1+ge9be34457` |
| PP | 2,048 |
| TG | 128 |
| Concurrency | 1 / 2 / 10 (Qwen3-8B Q4_K_M: 1 / 2 / 5 / 10) |
| Depth | 0 / 4,096 (Qwen3-8B Q4_K_M: 0 / 4,096 / 8,192 / 16,384 / 32,768) |
| Cache | Prefix |
| Latency | API |
| Runs | 1 warm-up + 3 measured; the 2026-11-16 Qwen3-30B-A3B, GPT-OSS, Gemma 3, Llama 3.1 and Mistral 7B references and the Mistral Small 4 reference: 1 warm-up + 2 measured |
| mirmir | 0.6.2, libmir `891a79b`, mircuda `fc3df90`, mirtal `0531428`, mirmir `9a0a73c` |
| Reference | llama.cpp `b1b770d` `llama-server -ngl 99 -fa on --kv-unified`, one slot per concurrent request, sharing a unified K/V pool: the capacity mirmir resolved for the model in this run, 360,448 tokens in the earlier references |
| K/V cache | mirmir BF16; llama.cpp its default F16, which on CUDA is faster than its BF16 (on four models measured both ways, llama.cpp's generation fell 9–30% with BF16 K/V) |
| Aggregation | Geometric mean over cells both engines completed |

| Model | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|:---|---:|---:|---:|
| [Qwen3-30B-A3B Q4_K_M](#qwen3-30b-a3b-q4_k_m) | 4,182.5 / 2,734.8<br>152.9% · 6/6 wins | 118.36 / 94.81<br>124.8% · 6/6 wins | 1,340 / 1,887<br>0.710× · 6/6 wins |
| [Qwen3-30B-A3B Q8_0](#qwen3-30b-a3b-q8_0) | 3,918.4 / 2,058.9<br>190.3% · 6/6 wins | 80.38 / 65.48<br>122.8% · 6/6 wins | 1,427 / 2,498<br>0.571× · 6/6 wins |
| [GPT-OSS-20B MXFP4](#gpt-oss-20b-mxfp4) | 5,752.8 / 3,693.3<br>155.8% · 6/6 wins | 135.04 / 112.04<br>120.5% · 5/6 wins | 1,003 / 1,471<br>0.682× · 6/6 wins |
| [Gemma 3 12B IT Q4_K_M](#gemma-3-12b-it-q4_k_m) | 2,595.5 / 1,427.3<br>181.9% · 6/6 wins | 48.48 / 40.98<br>118.3% · 6/6 wins | 1,890 / 3,443<br>0.549× · 6/6 wins |
| [Qwen3-8B Q4_K_M](#qwen3-8b-q4_k_m) | 2,782.0 / 1,119.4<br>248.5% · 20/20 wins | 61.61 / 40.34<br>152.7% · 17/20 wins | 2,319 / 5,069<br>0.458× · 20/20 wins |
| [Qwen3-8B Q8_0](#qwen3-8b-q8_0) | 3,638.4 / 2,194.9<br>165.8% · 6/6 wins | 56.82 / 47.79<br>118.9% · 5/6 wins | 1,556 / 2,402<br>0.648× · 6/6 wins |
| [Llama 3.1 8B Instruct Q4_K_M](#llama-31-8b-instruct-q4_k_m) | 3,880.0 / 2,662.6<br>145.7% · 6/6 wins | 75.93 / 65.58<br>115.8% · 6/6 wins | 1,256 / 1,793<br>0.701× · 6/6 wins |
| [Mistral 7B Instruct v0.3 Q4_K_M](#mistral-7b-instruct-v03-q4_k_m) | 4,059.8 / 2,644.3<br>153.5% · 6/6 wins | 79.78 / 68.90<br>115.8% · 6/6 wins | 1,202 / 1,829<br>0.657× · 6/6 wins |
| **CUDA only** | | | |
| [Qwen3.6-35B-A3B UD-Q4_K_M](#qwen36-35b-a3b-ud-q4_k_m) | 3,460.8 / 2,164.4<br>159.9% · 5/6 wins | 108.02 / 84.60<br>127.7% · 6/6 wins | 1,423 / 2,288<br>0.622× · 5/6 wins |
| [Gemma 4 26B-A4B IT UD-Q4_K_M](#gemma-4-26b-a4b-it-ud-q4_k_m) | 3,927.3 / 2,051.9<br>191.4% · 6/6 wins | 66.28 / 77.13<br>85.9% · 2/6 wins | 1,483 / 2,652<br>0.559× · 6/6 wins |
| [GLM-4.5-Air Q2_K](#glm-45-air-q2_k) | 1,229.2 / 758.0<br>162.2% · 2/2 wins | 38.53 / 37.09<br>103.9% · 2/2 wins | 2,357 / 3,858<br>0.611× · 2/2 wins |
| [Llama 4 Scout 17B-16E Q2_K](#llama-4-scout-17b-16e-q2_k) | 195.0 / 629.8<br>31.0% · 0/5 wins | 29.54 / 32.43<br>91.1% · 2/5 wins | 20,312 / 6,288<br>3.230× · 0/5 wins |
| [Nemotron Nano 9B v2 Q4_K_M](#nemotron-nano-9b-v2-q4_k_m) | 2,151.7 / 1,649.2<br>130.5% · 5/6 wins | 61.19 / 49.73<br>123.0% · 4/6 wins | 2,282 / 2,998<br>0.761× · 5/6 wins |
| [Mistral Small 4 119B UD-Q4_K_M](#mistral-small-4-119b-ud-q4_k_m) | 1,962.7 / 627.2<br>313.0% · 6/6 wins | 54.93 / 32.52<br>168.9% · 6/6 wins | 2,169 / 6,240<br>0.348× · 6/6 wins |

The first eight models are the [Metal page](gguf-metal.md)'s matrix; the
others run their GGUF files on CUDA only so far.

## Where mirmir trails

- **Llama 4 Scout prefill** is about half llama.cpp's at depth 0 and a sixth of
  it at depth 4,096. Scout's chunked-attention layers are the only ones mirmir's
  CUDA prefill cannot hand to FlashAttention, whose masks are causal or sliding;
  they run on the paged kernel instead.
- **Gemma 4 26B-A4B generation** at concurrency 1 and 2 is 25–30% below
  llama.cpp's; at concurrency 10 mirmir leads.
- **Hybrid models at depth 4,096 and concurrency 1:** Qwen3.6-35B-A3B's prefill
  falls from 4,419 to 1,662 tok/s and Nemotron Nano's to llama.cpp's level,
  while their prefill at depth 0 and at higher concurrency leads. Both keep
  recurrent state beside their attention layers.
- **GLM-4.5-Air and Llama 4 Scout at concurrency 10:** the capacity mirmir
  resolved left less than 12 GB of memory available once ten requests
  prefilled, and the harness's memory guard stopped the server; llama.cpp ran
  the same capacity. Those cells are `failed`.

## Quality

Perplexity over the wikitext-2 test set with llama.cpp's protocol
(`llama-perplexity`: 40 chunks of 512 tokens, scoring each chunk's second
half), on the same GGUF files and on this machine; Llama 4 Scout scores one
16,384-token chunk. Each chunk starts with BOS exactly where llama.cpp adds one
for the file's vocabulary. Lower is better; the two engines should agree to
within numerical noise.

| Model | mirmir | llama.cpp | Difference |
|:---|---:|---:|---:|
| Qwen3-8B Q4_K_M | 9.3462 | 9.3338 | +0.13% |
| Qwen3-8B Q2_K | 10.2718 | 10.2651 | +0.07% |
| Qwen3-8B UD-IQ2_M | 10.3560 | 10.3484 | +0.07% |
| Qwen3-30B-A3B Q4_K_M | 8.3170 | 8.2929 | +0.29% |
| Llama 3.1 8B Instruct Q4_K_M | 7.3135 | 7.3013 | +0.17% |
| Mistral 7B Instruct v0.3 Q4_K_M | 5.8244 | 5.8284 | −0.07% |
| Gemma 3 12B IT Q4_K_M | 8.5803 | 8.5619 | +0.21% |
| Llama 4 Scout Q2_K, 16,384-token context | 5.5786 | 5.5418 | +0.66% |
| Qwen3.6-35B-A3B UD-Q4_K_M | 5.8238 | 5.8217 | +0.04% |
| Qwen3.5-4B UD-Q2_K_XL | 10.2957 | 10.2990 | −0.03% |
| GLM-4.5-Air Q2_K | 4.8833 | 4.8840 | −0.01% |
| GLM-4 9B Q4_K_M | 9.1317 | 9.1282 | +0.04% |
| Nemotron Nano 9B v2 Q4_K_M | 7.8023 | 7.8019 | +0.01% |
| Gemma 2 9B IT Q4_K_M | 7.8431 | 7.8351 | +0.10% |
| Command R7B Q4_K_M | 7.3588 | 7.3559 | +0.04% |
| Phi-4 Q4_K_M | 6.1797 | 6.1728 | +0.11% |
| OLMo 2 7B Instruct Q4_K_M | 9.3888 | 9.3784 | +0.11% |
| Granite 3.1 3B-A800M MoE Q4_K_M | 7.4339 | 7.3882 | +0.62% |
| DeepSeek-V2-Lite Chat Q4_K_M | 8.6007 | 8.5475 | +0.62% |
| Moonlight 16B-A3B Instruct Q4_K_M | 8.7022 | 8.6807 | +0.25% |
| Mistral Small 4 119B UD-Q4_K_M | 5.0233 | 4.9962 | +0.54% |

The largest differences, Llama 4 Scout at 16,384 tokens, Granite 3.1 MoE,
DeepSeek-V2-Lite and Mistral Small 4 (+0.5% to +0.7%), are open: Scout's gap
grows with context length as on Metal (where it runs the other way), Granite
MoE varies between runs of the same build, and DeepSeek-V2-Lite does not while
Moonlight, with the same latent attention, differs by +0.25%.

## Qwen3-30B-A3B Q4_K_M

GGUF file `Qwen3-30B-A3B-Q4_K_M.gguf`; 6 of 6 cells completed on both engines; llama.cpp from `benchmarks/runs/2026-11-16-gguf-g4`.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 4,644.3 / 3,227.5 | 89.26 / 81.24 | 456 / 666 |
| 0 | 2 | 4,837.1 / 3,112.1 | 121.69 / 114.04 | 847 / 1,297 |
| 0 | 10 | 4,983.5 / 2,983.7 | 211.56 / 124.07 | 4,078 / 5,160 |
| 4,096 | 1 | 3,448.6 / 2,904.4 | 77.18 / 70.29 | 611 / 739 |
| 4,096 | 2 | 3,657.7 / 2,473.5 | 99.19 / 94.75 | 1,120 / 1,645 |
| 4,096 | 10 | 3,790.8 / 1,943.2 | 156.26 / 94.89 | 5,361 / 8,332 |

## Qwen3-30B-A3B Q8_0

GGUF file `Qwen3-30B-A3B-Q8_0.gguf`; 6 of 6 cells completed on both engines; llama.cpp from `benchmarks/runs/2026-11-16-gguf-g4`.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 4,281.2 / 2,368.5 | 59.83 / 55.23 | 501 / 909 |
| 0 | 2 | 4,529.6 / 2,250.3 | 81.78 / 73.69 | 904 / 1,793 |
| 0 | 10 | 4,721.1 / 2,227.0 | 135.26 / 86.84 | 4,229 / 6,873 |
| 4,096 | 1 | 3,176.3 / 2,148.6 | 53.57 / 48.30 | 670 / 1,000 |
| 4,096 | 2 | 3,446.1 / 1,925.5 | 69.80 / 64.80 | 1,188 / 2,111 |
| 4,096 | 10 | 3,611.7 / 1,551.1 | 109.03 / 71.24 | 5,534 / 10,285 |

## GPT-OSS-20B MXFP4

GGUF file `gpt-oss-20b-MXFP4.gguf`; 6 of 6 cells completed on both engines; llama.cpp from `benchmarks/runs/2026-11-16-gguf-g4`.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 6,279.9 / 3,982.8 | 77.23 / 76.04 | 371 / 558 |
| 0 | 2 | 6,336.6 / 3,704.3 | 114.25 / 107.61 | 646 / 1,097 |
| 0 | 10 | 6,571.6 / 3,953.7 | 315.09 / 191.46 | 3,106 / 4,526 |
| 4,096 | 1 | 4,739.2 / 3,840.5 | 73.80 / 74.94 | 479 / 579 |
| 4,096 | 2 | 5,119.1 / 3,496.4 | 109.70 / 105.47 | 800 / 1,163 |
| 4,096 | 10 | 5,713.6 / 3,240.4 | 269.43 / 159.74 | 3,570 / 5,420 |

## Gemma 3 12B IT Q4_K_M

GGUF file `gemma-3-12b-it-Q4_K_M.gguf`; 6 of 6 cells completed on both engines; llama.cpp from `benchmarks/runs/2026-11-16-gguf-g5`.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 2,621.1 / 1,637.2 | 27.69 / 25.21 | 784 / 1,253 |
| 0 | 2 | 2,463.8 / 1,668.9 | 50.02 / 44.62 | 1,823 / 2,399 |
| 0 | 10 | 2,881.3 / 1,520.6 | 94.66 / 77.49 | 4,844 / 9,763 |
| 4,096 | 1 | 2,423.2 / 1,638.0 | 26.37 / 24.61 | 848 / 1,252 |
| 4,096 | 2 | 2,730.5 / 1,374.3 | 46.81 / 40.00 | 1,501 / 2,724 |
| 4,096 | 10 | 2,483.4 / 903.8 | 80.27 / 55.16 | 5,178 / 16,656 |

## Qwen3-8B Q4_K_M

GGUF file `Qwen3-8B-Q4_K_M.gguf`; 20 of 20 cells completed on both engines; llama.cpp measured in this run.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 3,859.6 / 3,040.7 | 43.98 / 39.58 | 558 / 729 |
| 0 | 2 | 4,251.7 / 2,829.7 | 80.84 / 77.16 | 963 / 1,420 |
| 0 | 5 | 4,496.5 / 2,834.7 | 171.18 / 123.05 | 2,277 / 3,300 |
| 0 | 10 | 4,482.3 / 2,625.6 | 236.28 / 129.05 | 4,535 / 6,048 |
| 4,096 | 1 | 3,122.0 / 2,836.0 | 39.35 / 36.47 | 687 / 782 |
| 4,096 | 2 | 3,372.3 / 2,257.5 | 67.62 / 66.06 | 1,215 / 1,800 |
| 4,096 | 5 | 3,598.8 / 1,931.7 | 115.46 / 100.48 | 2,845 / 5,042 |
| 4,096 | 10 | 3,615.4 / 1,750.3 | 147.03 / 90.37 | 5,591 / 9,366 |
| 8,192 | 1 | 2,697.5 / 2,379.0 | 36.25 / 33.54 | 794 / 927 |
| 8,192 | 2 | 2,890.9 / 1,642.8 | 55.67 / 56.65 | 1,417 / 2,474 |
| 8,192 | 5 | 3,085.7 / 1,354.8 | 89.73 / 78.29 | 3,318 / 7,227 |
| 8,192 | 10 | 3,129.6 / 829.8 | 107.34 / 48.85 | 6,484 / 17,696 |
| 16,384 | 1 | 2,179.2 / 1,830.0 | 29.40 / 28.62 | 981 / 1,196 |
| 16,384 | 2 | 2,349.5 / 1,116.5 | 42.75 / 44.10 | 1,743 / 3,646 |
| 16,384 | 5 | 2,449.4 / 555.6 | 61.63 / 42.28 | 4,180 / 15,969 |
| 16,384 | 10 | 2,431.7 / 219.4 | 66.12 / 13.57 | 7,639 / 51,237 |
| 32,768 | 1 | 1,568.1 / 1,265.7 | 22.26 / 21.81 | 1,357 / 1,717 |
| 32,768 | 2 | 1,630.2 / 691.0 | 29.54 / 31.09 | 2,512 / 5,895 |
| 32,768 | 5 | 1,725.2 / 114.0 | 38.23 / 8.41 | 5,935 / 55,496 |
| 32,768 | 10 | 1,681.2 / 63.1 | 38.27 / 4.09 | 10,321 / 155,123 |

## Qwen3-8B Q8_0

GGUF file `Qwen3-8B-Q8_0.gguf`; 6 of 8 cells completed on both engines; llama.cpp from `benchmarks/runs/2026-11-16-gguf-g2`.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 3,825.1 / 2,637.9 | 26.84 / 26.52 | 576 / 858 |
| 0 | 2 | 4,020.8 / 2,431.8 | 51.21 / 50.29 | 1,019 / 1,643 |
| 0 | 5 | failed / 2,499.6 | failed / 88.66 | failed / 3,671 |
| 0 | 10 | 4,251.2 / 2,307.1 | 176.62 / 104.43 | 4,728 / 6,853 |
| 4,096 | 1 | 3,048.1 / 2,449.6 | 24.75 / 24.45 | 717 / 923 |
| 4,096 | 2 | 3,307.7 / 1,954.7 | 44.81 / 44.84 | 1,239 / 2,072 |
| 4,096 | 5 | missing / 1,722.8 | missing / 77.19 | missing / 5,637 |
| 4,096 | 10 | 3,519.3 / 1,578.1 | 125.08 / 77.98 | 5,756 / 10,406 |

## Llama 3.1 8B Instruct Q4_K_M

GGUF file `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`; 6 of 6 cells completed on both engines; llama.cpp from `benchmarks/runs/2026-11-16-gguf-g5`.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 3,893.4 / 3,081.7 | 44.66 / 39.98 | 528 / 666 |
| 0 | 2 | 4,236.0 / 3,092.4 | 82.23 / 78.87 | 968 / 1,309 |
| 0 | 10 | 4,658.4 / 2,871.1 | 162.57 / 118.40 | 2,926 / 4,250 |
| 4,096 | 1 | 3,148.8 / 2,917.2 | 40.33 / 36.69 | 652 / 703 |
| 4,096 | 2 | 3,669.9 / 2,406.1 | 70.24 / 67.21 | 1,116 / 1,685 |
| 4,096 | 10 | 3,843.1 / 1,855.2 | 113.34 / 86.44 | 3,609 / 7,586 |

## Mistral 7B Instruct v0.3 Q4_K_M

GGUF file `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`; 6 of 6 cells completed on both engines; llama.cpp from `benchmarks/runs/2026-11-16-gguf-g5`.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 3,972.3 / 3,067.7 | 47.70 / 42.43 | 517 / 669 |
| 0 | 2 | 4,488.9 / 3,009.1 | 87.57 / 84.23 | 913 / 1,349 |
| 0 | 10 | 4,767.5 / 2,910.7 | 167.65 / 122.81 | 2,875 / 4,486 |
| 4,096 | 1 | 3,456.9 / 2,894.7 | 42.78 / 39.08 | 594 / 709 |
| 4,096 | 2 | 3,792.3 / 2,413.9 | 73.24 / 70.84 | 1,080 / 1,681 |
| 4,096 | 10 | 4,017.4 / 1,820.7 | 117.51 / 88.02 | 3,458 / 7,749 |

## Qwen3.6-35B-A3B UD-Q4_K_M

GGUF file `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`; 6 of 6 cells completed on both engines; llama.cpp measured in this run.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 4,419.3 / 2,414.1 | 72.67 / 65.68 | 468 / 851 |
| 0 | 2 | 4,980.2 / 2,312.6 | 113.39 / 101.05 | 823 / 1,741 |
| 0 | 10 | 4,670.4 / 2,147.1 | 160.56 / 97.93 | 2,969 / 6,770 |
| 4,096 | 1 | 1,661.7 / 2,255.0 | 70.52 / 64.12 | 1,235 / 910 |
| 4,096 | 2 | 2,360.3 / 2,086.6 | 110.88 / 97.89 | 1,735 / 1,931 |
| 4,096 | 10 | 4,261.2 / 1,823.0 | 153.59 / 89.89 | 3,389 / 8,154 |

## Gemma 4 26B-A4B IT UD-Q4_K_M

GGUF file `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`; 6 of 6 cells completed on both engines; llama.cpp measured in this run.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 4,870.7 / 2,246.3 | 36.89 / 51.47 | 512 / 977 |
| 0 | 2 | 4,407.3 / 2,194.6 | 59.94 / 79.22 | 931 / 1,855 |
| 0 | 10 | 4,612.6 / 2,108.2 | 148.40 / 119.78 | 4,401 / 8,532 |
| 4,096 | 1 | 3,297.5 / 2,311.4 | 35.10 / 51.64 | 684 / 952 |
| 4,096 | 2 | 3,298.9 / 1,898.3 | 57.00 / 78.67 | 1,242 / 2,137 |
| 4,096 | 10 | 3,406.4 / 1,636.4 | 129.07 / 106.15 | 5,963 / 11,067 |

## GLM-4.5-Air Q2_K

GGUF file `GLM-4.5-Air-Q2_K.gguf`; 2 of 6 cells completed on both engines; llama.cpp measured in this run.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1,186.9 / 768.6 | 32.77 / 30.57 | 1,727 / 2,738 |
| 0 | 2 | 1,273.1 / 747.5 | 45.29 / 45.01 | 3,217 / 5,435 |
| 0 | 10 | failed / 693.0 | failed / 34.40 | failed / 22,209 |
| 4,096 | 1 | missing / 683.0 | missing / 26.53 | missing / 3,082 |
| 4,096 | 2 | missing / 615.1 | missing / 33.81 | missing / 6,628 |
| 4,096 | 10 | missing / 504.0 | missing / 26.90 | missing / 31,355 |

## Llama 4 Scout 17B-16E Q2_K

GGUF file `Llama-4-Scout-17B-16E-Instruct-Q2_K.gguf`; 5 of 6 cells completed on both engines; llama.cpp measured in this run.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 330.5 / 716.9 | 28.29 / 27.98 | 6,199 / 2,858 |
| 0 | 2 | 342.1 / 708.5 | 41.00 / 40.87 | 11,971 / 5,704 |
| 0 | 10 | 343.9 / 683.7 | 23.57 / 34.79 | 40,229 / 20,742 |
| 4,096 | 1 | 84.7 / 559.6 | 24.39 / 25.63 | 24,185 / 3,662 |
| 4,096 | 2 | 85.5 / 510.0 | 33.73 / 35.15 | 47,889 / 7,937 |
| 4,096 | 10 | failed / 262.9 | failed / 15.98 | failed / 50,114 |

## Nemotron Nano 9B v2 Q4_K_M

GGUF file `nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q4_K_M.gguf`; 6 of 6 cells completed on both engines; llama.cpp measured in this run.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 2,189.0 / 1,781.0 | 33.33 / 33.97 | 938 / 1,152 |
| 0 | 2 | 2,259.6 / 1,723.4 | 61.96 / 55.88 | 1,813 / 2,323 |
| 0 | 10 | 2,364.4 / 1,606.5 | 113.26 / 68.15 | 5,874 / 9,041 |
| 4,096 | 1 | 1,761.0 / 1,778.7 | 33.00 / 33.10 | 1,165 / 1,153 |
| 4,096 | 2 | 2,119.2 / 1,588.8 | 61.11 / 54.08 | 1,933 / 2,521 |
| 4,096 | 10 | 2,274.2 / 1,444.0 | 111.26 / 65.33 | 6,278 / 10,332 |

## Mistral Small 4 119B UD-Q4_K_M

GGUF file `Mistral-Small-4-119B-2603-UD-Q4_K_M-0000{1,2,3}-of-00003.gguf`; 6 of 6 cells completed on both engines; llama.cpp from `benchmarks/runs/2026-11-21-mistral4-gguf`.

| Depth | Concurrency | PP tok/s ↑<br>mirmir / llama.cpp | TG tok/s ↑<br>mirmir / llama.cpp | TTFT ms ↓<br>mirmir / llama.cpp |
|---:|---:|---:|---:|---:|
| 0 | 1 | 1,853.9 / 824.8 | 43.04 / 33.90 | 807 / 1,809 |
| 0 | 2 | 2,497.0 / 790.2 | 58.87 / 38.41 | 1,194 / 3,165 |
| 0 | 10 | 2,862.3 / 541.1 | 77.42 / 32.84 | 4,026 / 16,740 |
| 4,096 | 1 | 1,080.3 / 761.9 | 41.81 / 33.34 | 1,898 / 2,690 |
| 4,096 | 2 | 1,909.7 / 682.6 | 56.46 / 45.05 | 2,145 / 5,971 |
| 4,096 | 10 | 2,091.3 / 331.7 | 59.32 / 18.41 | 6,604 / 38,348 |

`failed`: the cell did not complete (llama-benchy stopped at its first error);
`missing`: the cell did not run after that failure.
