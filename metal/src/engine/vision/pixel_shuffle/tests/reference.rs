//! A host implementation of the pixel-shuffle encoder in `f64`, written from
//! the CUDA kernels: half-pixel bilinear positions, pre-norm blocks with exact
//! GELU, the fold over each block's rows then columns, RMS norm and `relu²`.
//! Plain arithmetic on purpose: it reads like the kernels it checks.
#![allow(clippy::suboptimal_flops)]

use models::{layout::PixelShuffleVisionConfig, vision::PixelShufflePreprocessedImage};

use super::{super::encoder, Weights};

type Matrix = Vec<Vec<f64>>;

fn real(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

fn tensor<'a>(weights: &'a Weights, name: &str) -> &'a [f32] {
    weights.get(name).map_or(&[], |(_, data)| data.as_slice())
}

fn linear(input: &Matrix, weights: &Weights, prefix: &str, outputs: usize) -> Matrix {
    let weight = tensor(weights, &format!("{prefix}.weight"));
    let bias = tensor(weights, &format!("{prefix}.bias"));
    input
        .iter()
        .map(|row| {
            (0..outputs)
                .map(|out| {
                    let dot = row
                        .iter()
                        .enumerate()
                        .map(|(i, x)| x * f64::from(weight[out * row.len() + i]))
                        .sum::<f64>();
                    dot + bias.get(out).map_or(0.0, |b| f64::from(*b))
                })
                .collect()
        })
        .collect()
}

fn layer_norm(input: &Matrix, weights: &Weights, prefix: &str, eps: f64) -> Matrix {
    let (weight, bias) = (
        tensor(weights, &format!("{prefix}.weight")),
        tensor(weights, &format!("{prefix}.bias")),
    );
    input
        .iter()
        .map(|row| {
            let mean = row.iter().sum::<f64>() / real(row.len());
            let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / real(row.len());
            row.iter()
                .enumerate()
                .map(|(i, x)| {
                    (x - mean) / (var + eps).sqrt() * f64::from(weight[i]) + f64::from(bias[i])
                })
                .collect()
        })
        .collect()
}

/// Abramowitz and Stegun 7.1.26, within 1.5e-7.
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    (1.0 - poly * (-x * x).exp()).copysign(x)
}

fn add(left: &Matrix, right: &Matrix) -> Matrix {
    left.iter()
        .zip(right)
        .map(|(l, r)| l.iter().zip(r).map(|(a, b)| a + b).collect())
        .collect()
}

fn attention(input: &Matrix, heads: usize) -> Matrix {
    let width = input[0].len() / 3;
    let head_dim = width / heads;
    let scale = 1.0 / real(head_dim).sqrt();
    input
        .iter()
        .map(|query_row| {
            (0..heads)
                .flat_map(|head| {
                    let base = head * head_dim;
                    let scores = input
                        .iter()
                        .map(|key_row| {
                            (0..head_dim)
                                .map(|d| query_row[base + d] * key_row[width + base + d])
                                .sum::<f64>()
                                * scale
                        })
                        .collect::<Vec<_>>();
                    let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exps = scores.iter().map(|s| (s - peak).exp()).collect::<Vec<_>>();
                    let total = exps.iter().sum::<f64>();
                    (0..head_dim)
                        .map(|d| {
                            input
                                .iter()
                                .zip(&exps)
                                .map(|(row, e)| e / total * row[2 * width + base + d])
                                .sum::<f64>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        })
        .collect()
}

fn embed(
    config: &PixelShuffleVisionConfig,
    weights: &Weights,
    image: &PixelShufflePreprocessedImage,
) -> Matrix {
    let (h, grid) = (config.hidden_size, config.position_grid);
    let (rows, cols) = (image.grid_height, image.grid_width);
    let width = image.patches.len() / (rows * cols);
    let patches = image
        .patches
        .chunks(width)
        .map(|patch| patch.iter().map(|x| f64::from(*x)).collect())
        .collect::<Matrix>();
    let projected = linear(&patches, weights, &encoder("patch_generator.embedder"), h);
    let table = tensor(weights, &encoder("patch_generator.pos_embed"));
    let scale = real(grid) / real(rows.max(cols));
    let texel = |index: usize| {
        let source = ((real(index) + 0.5) * scale - 0.5).max(0.0);
        let lower = (0..grid).rev().find(|t| real(*t) <= source).unwrap_or(0);
        (lower, (lower + 1).min(grid - 1), source - real(lower))
    };
    let at = |cell: usize, f: usize| f64::from(table[cell * h + f]);
    let prefix = tensor(weights, &encoder("patch_generator.cls_token.token"));
    let mut tokens = prefix
        .chunks(h)
        .map(|row| row.iter().map(|x| f64::from(*x)).collect())
        .collect::<Matrix>();
    for (patch, row) in projected.iter().enumerate() {
        let ((y0, y1, dy), (x0, x1, dx)) = (texel(patch / cols), texel(patch % cols));
        tokens.push(
            (0..h)
                .map(|f| {
                    let top = (1.0 - dx) * at(y0 * grid + x0, f) + dx * at(y0 * grid + x1, f);
                    let bottom = (1.0 - dx) * at(y1 * grid + x0, f) + dx * at(y1 * grid + x1, f);
                    row[f] + (1.0 - dy) * top + dy * bottom
                })
                .collect(),
        );
    }
    tokens
}

pub(super) fn tower(
    config: &PixelShuffleVisionConfig,
    weights: &Weights,
    image: &PixelShufflePreprocessedImage,
) -> Vec<f32> {
    let h = config.hidden_size;
    let mut hidden = embed(config, weights, image);
    for layer in 0..config.num_hidden_layers {
        let name = |suffix: &str| encoder(&format!("blocks.{layer}.{suffix}"));
        let qkv = linear(
            &layer_norm(&hidden, weights, &name("norm1"), config.layer_norm_eps),
            weights,
            &name("attn.qkv"),
            3 * h,
        );
        hidden = add(
            &hidden,
            &linear(&attention(&qkv, config.num_attention_heads), weights, &name("attn.proj"), h),
        );
        let up = linear(
            &layer_norm(&hidden, weights, &name("norm2"), config.layer_norm_eps),
            weights,
            &name("mlp.fc1"),
            config.intermediate_size,
        );
        let activated = up
            .iter()
            .map(|row| row.iter().map(|x| 0.5 * x * (1.0 + erf(x / 2.0_f64.sqrt()))).collect())
            .collect::<Matrix>();
        hidden = add(&hidden, &linear(&activated, weights, &name("mlp.fc2"), h));
    }
    let encoded = layer_norm(
        &hidden,
        weights,
        "vision_projector.vision_final_layernorm",
        config.layer_norm_eps,
    );
    let (f, cols) = (config.shuffle_factor, image.grid_width);
    let folded = (0..image.soft_tokens)
        .map(|token| {
            (0..f * f * h)
                .map(|channel| {
                    let block = channel / h;
                    let row = (token / (cols / f)) * f + block / f;
                    let col = (token % (cols / f)) * f + block % f;
                    encoded[config.prefix_tokens + row * cols + col][channel % h]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Matrix>();
    let norm = tensor(weights, "mlp1.0.weight");
    let normalized = folded
        .iter()
        .map(|row| {
            let rms = (row.iter().map(|x| x * x).sum::<f64>() / real(row.len())
                + config.projector_norm_eps)
                .sqrt();
            row.iter().enumerate().map(|(i, x)| x / rms * f64::from(norm[i])).collect()
        })
        .collect::<Matrix>();
    let first = linear(&normalized, weights, "mlp1.1", config.projector_hidden_size)
        .into_iter()
        .map(|row| row.into_iter().map(|x| x.max(0.0).powi(2)).collect())
        .collect::<Matrix>();
    linear(&first, weights, "mlp1.3", config.output_hidden_size)
        .into_iter()
        .flatten()
        .map(|x| x.to_string().parse::<f32>().unwrap_or(f32::NAN))
        .collect()
}
