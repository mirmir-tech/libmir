//! FP64 host reference of one Mamba-2 mixer, rounding to BF16 at the same
//! points as the kernels: convolution output, scan output, normalized value,
//! and the weighted result.

use mircuda::bf16;

use crate::kernels::Mamba2Geometry;

pub(super) struct Parameters {
    pub(super) convolution_weight: Vec<bf16>,
    pub(super) convolution_bias: Vec<bf16>,
    pub(super) a_log: Vec<bf16>,
    pub(super) skip: Vec<bf16>,
    pub(super) dt_bias: Vec<bf16>,
    pub(super) norm_weight: Vec<bf16>,
}

pub(super) struct Reference {
    geometry: Mamba2Geometry,
    pub(super) state: Vec<f64>,
    history: Vec<f64>,
}

fn round(value: f64) -> f64 {
    f64::from(bf16::from_f64(value).to_f32())
}

fn silu(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

fn softplus(value: f64) -> f64 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

pub(super) fn count(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

fn value(values: &[bf16], index: usize) -> f64 {
    f64::from(values[index].to_f32())
}

impl Reference {
    pub(super) fn new(geometry: Mamba2Geometry) -> Self {
        let inner = geometry.heads * geometry.head_dim;
        let channels = inner + 2 * geometry.groups * geometry.state_size;
        Self {
            geometry,
            state: vec![0.0; inner * geometry.state_size],
            history: vec![0.0; (geometry.convolution_kernel_size - 1) * channels],
        }
    }

    /// Advances by `tokens` projection rows and returns the normalized output.
    pub(super) fn execute(&mut self, projection: &[bf16], parameters: &Parameters) -> Vec<f64> {
        let geometry = self.geometry;
        let (heads, head_dim, state_size) =
            (geometry.heads, geometry.head_dim, geometry.state_size);
        let inner = heads * head_dim;
        let channels = inner + 2 * geometry.groups * state_size;
        let width = inner + channels + heads;
        let kernel = geometry.convolution_kernel_size;
        let tokens = projection.len() / width;
        let mut combined = self.history.clone();
        for token in 0..tokens {
            combined.extend(
                (0..channels).map(|channel| value(projection, token * width + inner + channel)),
            );
        }
        let mut output = Vec::with_capacity(tokens * inner);
        for token in 0..tokens {
            let convolved: Vec<f64> = (0..channels)
                .map(|channel| {
                    let sum = (0..kernel).fold(
                        value(&parameters.convolution_bias, channel),
                        |sum, tap| {
                            combined[(token + tap) * channels + channel].mul_add(
                                value(&parameters.convolution_weight, channel * kernel + tap),
                                sum,
                            )
                        },
                    );
                    round(silu(sum))
                })
                .collect();
            let mut scanned = vec![0.0; inner];
            for head in 0..heads {
                let group = head / (heads / geometry.groups);
                let raw = value(projection, token * width + inner + channels + head);
                let dt = softplus(raw + value(&parameters.dt_bias, head)).clamp(
                    f64::from(geometry.time_step_minimum),
                    f64::from(geometry.time_step_maximum),
                );
                let decay = (dt * -value(&parameters.a_log, head).exp()).exp();
                for row in 0..head_dim {
                    let x = convolved[head * head_dim + row];
                    let mut sum = 0.0;
                    for column in 0..state_size {
                        let b = convolved[inner + group * state_size + column];
                        let c = convolved[inner + (geometry.groups + group) * state_size + column];
                        let slot = &mut self.state[(head * head_dim + row) * state_size + column];
                        *slot = (dt * x).mul_add(b, *slot * decay);
                        sum = slot.mul_add(c, sum);
                    }
                    scanned[head * head_dim + row] =
                        round(value(&parameters.skip, head).mul_add(x, sum));
                }
            }
            let group_size = inner / geometry.groups;
            for group in 0..geometry.groups {
                let gated: Vec<f64> = (0..group_size)
                    .map(|index| {
                        let channel = group * group_size + index;
                        scanned[channel] * silu(value(projection, token * width + channel))
                    })
                    .collect();
                let mean = gated.iter().map(|value| value * value).sum::<f64>() / count(group_size);
                let scale = 1.0 / (mean + f64::from(geometry.norm_epsilon)).sqrt();
                output.extend(gated.iter().enumerate().map(|(index, gated)| {
                    round(
                        round(gated * scale)
                            * value(&parameters.norm_weight, group * group_size + index),
                    )
                }));
            }
        }
        let keep = (kernel - 1) * channels;
        self.history = combined[combined.len() - keep..].to_vec();
        output
    }
}
