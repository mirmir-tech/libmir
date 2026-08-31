use mircuda::{DeviceBuffer, bf16, cuda_export};

cuda_export!(pub(super) CandidatesKernel = "libmir_cuda_sampling_candidates_bf16"(
    logits: &DeviceBuffer<bf16>, output: &mut DeviceBuffer<u64>,
    denominator: &mut DeviceBuffer<f32>, vocab: u32, logits_stride: u32, top_k: u32,
    row: u32, workspace_stride: u32,
));
cuda_export!(pub(super) MergeKernel = "libmir_cuda_sampling_merge"(
    input: &DeviceBuffer<u64>, output: &mut DeviceBuffer<u64>,
    denominator: &mut DeviceBuffer<f32>, count: u32, top_k: u32,
    row: u32, workspace_stride: u32,
));
cuda_export!(pub(super) MassKernel = "libmir_cuda_sampling_mass_bf16"(
    logits: &DeviceBuffer<bf16>, candidates: &DeviceBuffer<u64>,
    denominator: &mut DeviceBuffer<f32>, vocab: u32, logits_stride: u32, row: u32,
    workspace_stride: u32,
));
cuda_export!(pub(super) FinalizeKernel = "libmir_cuda_sampling_finalize_bf16"(
    logits: &DeviceBuffer<bf16>, candidates: &DeviceBuffer<u64>,
    denominator: &DeviceBuffer<f32>, output: &mut DeviceBuffer<u32>,
    top_k: u32, top_p: f32, temperature: f32, draw: f32,
    vocab: u32, logits_stride: u32, row: u32, workspace_stride: u32,
));
cuda_export!(pub(super) FullMassKernel = "libmir_cuda_sampling_full_mass_bf16"(
    logits: &DeviceBuffer<bf16>, candidates: &DeviceBuffer<u64>,
    block_mass: &mut DeviceBuffer<f32>, vocab: u32, temperature: f32,
    logits_stride: u32, row: u32, workspace_stride: u32,
));
cuda_export!(pub(super) FullFinalizeKernel = "libmir_cuda_sampling_full_finalize_bf16"(
    logits: &DeviceBuffer<bf16>, candidates: &DeviceBuffer<u64>,
    block_mass: &DeviceBuffer<f32>, output: &mut DeviceBuffer<u32>,
    temperature: f32, draw: f32, vocab: u32, logits_stride: u32, row: u32,
    workspace_stride: u32, block_count: u32,
));
