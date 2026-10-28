use mircuda::{
    CompileOptions, Compiler, DeviceBuffer, TypedKernel, bf16, cuda_export, cuda_kernel_file,
};

use super::DecisionWindow;
use crate::Result;

cuda_export!(pub(super) NormF32 = "libmir_decision_layer_norm_f32"(
    input: &DeviceBuffer<f32>, weight: &DeviceBuffer<f32>, bias: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>, rows: u32, width: u32, epsilon: f32,
));
cuda_export!(pub(super) NormBf16 = "libmir_decision_layer_norm_bf16"(
    input: &DeviceBuffer<f32>, weight: &DeviceBuffer<f32>, bias: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<bf16>, rows: u32, width: u32, epsilon: f32,
));
cuda_export!(pub(super) GegluF32 = "libmir_decision_geglu_f32"(
    input: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>, rows: u32, width: u32,
));
cuda_export!(pub(super) GegluBf16 = "libmir_decision_geglu_bf16"(
    input: &DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>, rows: u32, width: u32,
));
cuda_export!(pub(super) SplitRopeF32 = "libmir_decision_split_rope_f32"(
    qkv: &DeviceBuffer<f32>, cosines: &DeviceBuffer<f32>, sines: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>, tokens: u32, length: u32, heads: u32, dim: u32, pad: u32,
));
cuda_export!(pub(super) SplitRopeBf16 = "libmir_decision_split_rope_bf16"(
    qkv: &DeviceBuffer<f32>, cosines: &DeviceBuffer<f32>, sines: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<bf16>, tokens: u32, length: u32, heads: u32, dim: u32, pad: u32,
));
cuda_export!(pub(super) RotateF32 = "libmir_decision_rotate_f32"(
    qkv: &DeviceBuffer<f32>, cosines: &DeviceBuffer<f32>, sines: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>, tokens: u32, length: u32, heads: u32, dim: u32,
));
cuda_export!(pub(super) RotateBf16 = "libmir_decision_rotate_bf16"(
    qkv: &DeviceBuffer<f32>, cosines: &DeviceBuffer<f32>, sines: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<bf16>, tokens: u32, length: u32, heads: u32, dim: u32,
));
cuda_export!(pub(super) MergeF32 = "libmir_decision_merge_f32"(
    input: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>, tokens: u32, heads: u32,
    dim: u32,
));
cuda_export!(pub(super) MergeBf16 = "libmir_decision_merge_bf16"(
    input: &DeviceBuffer<f32>, output: &mut DeviceBuffer<bf16>, tokens: u32, heads: u32,
    dim: u32,
));
cuda_export!(pub(super) SoftmaxF32 = "libmir_decision_softmax_f32"(
    scores: &DeviceBuffer<f32>, probabilities: &mut DeviceBuffer<f32>,
    lengths: &DeviceBuffer<u32>, queries: u32, keys: u32, groups: u32, shift: u32, length: u32,
    radius: u32,
));
cuda_export!(pub(super) SoftmaxBf16 = "libmir_decision_softmax_bf16"(
    scores: &DeviceBuffer<f32>, probabilities: &mut DeviceBuffer<bf16>,
    lengths: &DeviceBuffer<u32>, queries: u32, keys: u32, groups: u32, shift: u32, length: u32,
    radius: u32,
));

/// Kernels that write the input of the next matrix product in f32 or bf16.
#[derive(Clone, Debug)]
pub struct DecisionLayout {
    pub(super) norm: (TypedKernel<NormF32>, TypedKernel<NormBf16>),
    pub(super) geglu: (TypedKernel<GegluF32>, TypedKernel<GegluBf16>),
    pub(super) split_rope: (TypedKernel<SplitRopeF32>, TypedKernel<SplitRopeBf16>),
    pub(super) rotate: (TypedKernel<RotateF32>, TypedKernel<RotateBf16>),
    pub(super) merge: (TypedKernel<MergeF32>, TypedKernel<MergeBf16>),
    pub(super) softmax: (TypedKernel<SoftmaxF32>, TypedKernel<SoftmaxBf16>),
}

impl DecisionLayout {
    pub fn compile(compiler: &Compiler) -> Result<Self> {
        let options = CompileOptions::default();
        let layout =
            compiler.compile(cuda_kernel_file!("../../../kernels/decision/layout.cu"), &options)?;
        let softmax = compiler
            .compile(cuda_kernel_file!("../../../kernels/decision/softmax.cu"), &options)?;
        Ok(Self {
            norm: (layout.kernel()?, layout.kernel()?),
            geglu: (layout.kernel()?, layout.kernel()?),
            split_rope: (layout.kernel()?, layout.kernel()?),
            rotate: (layout.kernel()?, layout.kernel()?),
            merge: (layout.kernel()?, layout.kernel()?),
            softmax: (softmax.kernel()?, softmax.kernel()?),
        })
    }
}

/// Rotate-half `RoPE` angles, `[positions][head_dim / 2]`.
pub struct RopeTables {
    pub cosines: DeviceBuffer<f32>,
    pub sines: DeviceBuffer<f32>,
}

/// Head-major operands `[pad][queries | keys | values][pad]` of `tokens`
/// flattened tokens, `length` per sequence.
#[derive(Clone, Copy, Debug)]
pub struct HeadLayout {
    pub tokens: usize,
    pub length: usize,
    pub heads: usize,
    pub dim: usize,
    pub pad: usize,
}

impl HeadLayout {
    #[must_use]
    pub const fn elements(self) -> usize {
        (3 * self.heads * self.tokens + 2 * self.pad) * self.dim
    }

    /// Element offset of the first query, key or value of the first head.
    #[must_use]
    pub const fn part(self, part: usize) -> usize {
        (self.pad + part * self.heads * self.tokens) * self.dim
    }
}

/// Score rows of one strided batched product: member `i` holds `queries`
/// consecutive queries of group `i % groups` and `keys` keys starting `shift`
/// tokens before them.
#[derive(Clone, Copy, Debug)]
pub struct ScoreRows<'a> {
    pub lengths: &'a DeviceBuffer<u32>,
    pub queries: usize,
    pub keys: usize,
    pub groups: usize,
    pub shift: usize,
    pub length: usize,
    pub window: DecisionWindow,
}
