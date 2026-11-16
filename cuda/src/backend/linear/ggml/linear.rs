use std::sync::{Arc, Mutex};

use mircuda::{DeviceBuffer, Stream, bf16};

use super::GgmlCheckpointWeight;
use crate::{
    Bf16Projection, CudaBackend, CudaTensor, DensePlanRequest, DenseRole, Error, ExecutionPhase,
    Result,
    kernels::{
        GGML_GEMV_MAX_TOKENS, GgmlActivations, GgmlDequantize, GgmlKernelType, GgmlLinear, GgmlSpec,
    },
};

/// Every GGML type with kernels. One plan serves a projection in every
/// layer, and a `Q4_K_M` checkpoint stores the same projection as `Q4_K` in
/// some layers and `Q6_K` in others, so the weight's type picks the kernel.
const KINDS: [GgmlKernelType; 20] = GgmlKernelType::ALL;

/// Prepared BF16-input projection over a late-bound GGML weight: the GEMV
/// for decode batches, dequantization into shared BF16 scratch followed by
/// the tuned dense GEMM for prefill.
///
/// Dequantizing to BF16 costs 0.2-0.3% wikitext-2 perplexity on 5- and 8-bit
/// types against llama.cpp's int8 path; FP16 halved that but cost 19-30%
/// prefill throughput. An int8 tensor-core prefill would remove both.
#[derive(Debug)]
pub struct GgmlBf16Linear {
    path: Path,
    input_features: usize,
    output_features: usize,
    stream: Stream,
}

#[derive(Debug)]
enum Path {
    Vector {
        operations: Vec<GgmlLinear>,
        activations: Arc<Mutex<GgmlActivations>>,
    },
    Dense {
        dequantize: Vec<GgmlDequantize>,
        projection: Mutex<Bf16Projection>,
        scratch: Arc<Mutex<DeviceBuffer<bf16>>>,
    },
}

impl GgmlBf16Linear {
    pub(super) fn new(
        backend: &CudaBackend,
        tokens: usize,
        role: DenseRole,
        input_features: usize,
        output_features: usize,
    ) -> Result<Self> {
        let compiler = &backend.inner.compiler;
        // A type whose blocks do not tile the input width cannot store this
        // projection, so it gets no kernel.
        let specs = KINDS
            .into_iter()
            .filter_map(|kind| GgmlSpec::new(kind, tokens, input_features, output_features).ok())
            .collect::<Vec<_>>();
        if specs.is_empty() {
            return Err(Error::InvalidDecoderKernel("invalid GGML projection geometry"));
        }
        let path = if tokens <= GGML_GEMV_MAX_TOKENS {
            let operations = specs.into_iter().map(|spec| GgmlLinear::compile(compiler, spec));
            Path::Vector {
                operations: operations.collect::<Result<_>>()?,
                activations: backend.ggml_activations(tokens, input_features)?,
            }
        } else {
            let dequantize = specs
                .into_iter()
                .map(|spec| {
                    GgmlDequantize::compile(compiler, spec.kind, output_features, input_features)
                })
                .collect::<Result<Vec<_>>>()?;
            let projection = backend.prepare_bf16_projection(DensePlanRequest {
                phase: ExecutionPhase::Prefill,
                role,
                tokens,
                input_features,
                output_features,
            })?;
            let elements = output_features
                .checked_mul(input_features)
                .ok_or(Error::InvalidDecoderKernel("GGML scratch size overflow"))?;
            Path::Dense {
                scratch: backend.ggml_dequantize_scratch(elements)?,
                dequantize,
                projection: Mutex::new(projection),
            }
        };
        Ok(Self {
            path,
            input_features,
            output_features,
            stream: backend.inner.stream.clone(),
        })
    }

    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &GgmlCheckpointWeight,
        output: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        self.execute_shared(input, weight, output, false)
    }

    /// [`Self::execute`] for one of several projections of one input, which
    /// share the quantized activations of plans with the same batch and
    /// width: `quantized` says an earlier projection of this input on the
    /// stream already wrote them.
    pub fn execute_shared(
        &self,
        input: &DeviceBuffer<bf16>,
        weight: &GgmlCheckpointWeight,
        output: &mut DeviceBuffer<bf16>,
        quantized: bool,
    ) -> Result<()> {
        if self.input_features != weight.input_features
            || self.output_features != weight.output_features
        {
            return Err(Error::InvalidExecutionPlan(
                "GGML plan and late-bound weight geometry differ",
            ));
        }
        let missing = || Error::InvalidExecutionPlan("GGML plan has no kernel for the weight type");
        match &self.path {
            Path::Vector { operations, activations } => {
                let operation = operations
                    .iter()
                    .find(|operation| operation.spec().kind == weight.kind)
                    .ok_or_else(missing)?;
                let mut activations = activations
                    .lock()
                    .map_err(|_| Error::InvalidExecutionPlan("GGML activation lock is poisoned"))?;
                if !quantized {
                    operation.quantize(&self.stream, input, &mut activations)?;
                }
                operation.multiply(&self.stream, &activations, &weight.blocks, output)
            },
            Path::Dense { dequantize, projection, scratch } => {
                let dequantize = dequantize
                    .iter()
                    .find(|operation| operation.kind() == weight.kind)
                    .ok_or_else(missing)?;
                let poisoned = || Error::InvalidExecutionPlan("GGML scratch lock is poisoned");
                // The stream orders this projection's use of the scratch
                // before the next one's; the lock only guards the handle.
                let buffer = {
                    let mut scratch = scratch.lock().map_err(|_| poisoned())?;
                    dequantize.execute(&self.stream, &weight.blocks, &mut scratch)?;
                    scratch.clone()
                };
                let dense = CudaTensor::from_bf16(
                    "GGML dequantized projection".into(),
                    vec![self.output_features, self.input_features],
                    buffer,
                );
                projection.lock().map_err(|_| poisoned())?.execute(input, &dense, output)
            },
        }
    }
}
