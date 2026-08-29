use mircuda::{DeviceBuffer, bf16, cuda_export};

cuda_export!(pub(super) LogParameters = "libmir_cuda_gated_delta_log_parameters_bf16"(
    alpha: &DeviceBuffer<bf16>, beta: &DeviceBuffer<bf16>,
    a_log: &DeviceBuffer<bf16>, dt_bias: &DeviceBuffer<bf16>,
    log_decay: &mut DeviceBuffer<f32>, update: &mut DeviceBuffer<f32>,
    tokens: u32, value_heads: u32,
));

cuda_export!(pub(super) Cumsum = "chunk_local_cumsum_scalar_kernel"(
    input: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>,
    cu_seqlens: &DeviceBuffer<i32>, chunk_indices: &DeviceBuffer<i32>, tokens: u32,
    global_scratch: u64, profile_scratch: u64,
));

cuda_export!(pub(super) Kkt = "chunk_scaled_dot_kkt_fwd_kernel"(
    key: &DeviceBuffer<bf16>, beta: &DeviceBuffer<f32>, gate: &DeviceBuffer<f32>,
    matrix: &mut DeviceBuffer<f32>, cu_seqlens: &DeviceBuffer<i32>,
    chunk_indices: &DeviceBuffer<i32>, tokens: u32,
    global_scratch: u64, profile_scratch: u64,
));

cuda_export!(pub(super) Solve = "merge_16x16_to_64x64_inverse_kernel"(
    matrix: &DeviceBuffer<f32>, inverse: &mut DeviceBuffer<bf16>,
    cu_seqlens: &DeviceBuffer<i32>, chunk_indices: &DeviceBuffer<i32>, tokens: u32,
    global_scratch: u64, profile_scratch: u64,
));

cuda_export!(pub(super) Uw = "recompute_w_u_fwd_kernel"(
    key: &DeviceBuffer<bf16>, value: &DeviceBuffer<bf16>, beta: &DeviceBuffer<f32>,
    w: &mut DeviceBuffer<bf16>, u: &mut DeviceBuffer<bf16>,
    inverse: &DeviceBuffer<bf16>, gate: &DeviceBuffer<f32>,
    cu_seqlens: &DeviceBuffer<i32>, chunk_indices: &DeviceBuffer<i32>, tokens: u32,
    global_scratch: u64, profile_scratch: u64,
));

cuda_export!(pub(super) H = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"(
    key: &DeviceBuffer<bf16>, u: &DeviceBuffer<bf16>, w: &DeviceBuffer<bf16>,
    value: &mut DeviceBuffer<bf16>, gate: &DeviceBuffer<f32>,
    chunks: &mut DeviceBuffer<bf16>, initial_state: &DeviceBuffer<f32>,
    final_state: &mut DeviceBuffer<f32>, cu_seqlens: &DeviceBuffer<i32>,
    chunk_offsets: &DeviceBuffer<i32>, tokens: u32,
    global_scratch: u64, profile_scratch: u64,
));

cuda_export!(pub(super) O = "chunk_fwd_kernel_o"(
    query: &DeviceBuffer<bf16>, key: &DeviceBuffer<bf16>, value: &DeviceBuffer<bf16>,
    chunks: &DeviceBuffer<bf16>, gate: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<bf16>, cu_seqlens: &DeviceBuffer<i32>,
    chunk_indices: &DeviceBuffer<i32>, scale: f32, tokens: u32,
    global_scratch: u64, profile_scratch: u64,
));
