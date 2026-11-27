//! The storage of one clamped-routed layer's expert banks, classified as the
//! accelerator loaders dispatch on it: CUDA's `ClampedRoutedLayout` and
//! Metal's `ClampedRoutedExperts::load`.

use models::{
    gguf::GgmlType,
    weights::{
        BlockFormat, BlockProjectionLayout, ExpertProjectionRole, GgmlCompanion,
        RoutedExpertBindings, TensorBinding, TensorStorage,
    },
};

use crate::{EncodingKind, model::admission::WeightEncoding};

/// One layer's gate, up and down banks in one storage a loader reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ExpertBanks {
    /// Separate GGML MXFP4 banks, each with its bias (GGUF).
    GgmlMxFp4,
    /// Interleaved gate/up and down OCP MXFP4 banks with scales and biases
    /// (`OpenAI`'s `SafeTensors`).
    InterleavedMxFp4,
    /// Separate grouped-affine banks with output biases (MLX).
    Affine,
    /// Dense banks, separate or with fused gate/up.
    Dense,
    /// Block-quantized banks of one format outside the OCP MXFP4 layout,
    /// which Metal binds projection by projection.
    Block(BlockFormat),
}

/// Whether Metal's loader composes these banks projection by projection:
/// its separate and fused branches bind each projection on its own, so
/// any mix of dense and block-quantized banks loads. GGML and affine banks
/// load only as one uniform layout, and CUDA reads none mixed.
pub(super) fn metal_composes(experts: RoutedExpertBindings<'_>) -> bool {
    let projections = match experts {
        RoutedExpertBindings::InterleavedGateUp { gate_up, down } => vec![gate_up, down],
        RoutedExpertBindings::SeparateGateUp { gate, up, down } => vec![gate, up, down],
        RoutedExpertBindings::UngatedStacked { .. } | RoutedExpertBindings::Individual { .. } => {
            return false;
        },
    };
    projections.iter().all(|binding| {
        matches!(
            binding.storage,
            TensorStorage::Dense { .. } | TensorStorage::BlockQuantized { .. }
        )
    })
}

/// The layout CUDA's clamped-routed loader selects for the whole model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CudaExpertLayout {
    Ggml,
    Native,
    Mlx,
    Dense,
}

/// Why a layer's banks are in no storage a loader reads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum BankRefusal {
    /// The projections are stored in different encoding families.
    Mixed(Vec<(ExpertProjectionRole, WeightEncoding)>),
    /// GGML banks of a type other than MXFP4.
    GgmlType(GgmlType),
    /// Banks without the output bias every clamped-routed loader adds.
    Unbiased(WeightEncoding),
    /// An encoding no clamped-routed loader reads in this arrangement.
    Unread(WeightEncoding),
    /// Experts bound one by one, without a gate, or as auxiliary tensors,
    /// with the family of the first.
    Unstacked(EncodingKind),
}

impl ExpertBanks {
    /// The banks `experts` are stored in.
    pub(super) fn of(experts: RoutedExpertBindings<'_>) -> Result<Self, BankRefusal> {
        let (interleaved, projections) = match experts {
            RoutedExpertBindings::InterleavedGateUp { gate_up, down } => (
                true,
                vec![(ExpertProjectionRole::GateUp, gate_up), (ExpertProjectionRole::Down, down)],
            ),
            RoutedExpertBindings::SeparateGateUp { gate, up, down } => (
                false,
                vec![
                    (ExpertProjectionRole::Gate, gate),
                    (ExpertProjectionRole::Up, up),
                    (ExpertProjectionRole::Down, down),
                ],
            ),
            RoutedExpertBindings::UngatedStacked { up, .. } => {
                return Err(BankRefusal::Unstacked(family(up)));
            },
            RoutedExpertBindings::Individual { .. } => {
                let first = experts.individual(ExpertProjectionRole::Gate);
                return Err(BankRefusal::Unstacked(
                    first.first().map_or(EncodingKind::Dense, |binding| family(binding)),
                ));
            },
        };
        let encodings = projections
            .iter()
            .map(|(role, binding)| {
                WeightEncoding::of(&binding.storage).map(|encoding| (*role, encoding))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(BankRefusal::Unstacked(EncodingKind::Dense))?;
        let kind = encodings[0].1.kind();
        if encodings.iter().any(|(_, encoding)| encoding.kind() != kind) {
            return Err(BankRefusal::Mixed(encodings));
        }
        let first = encodings[0].1.clone();
        let bindings = projections.iter().map(|(_, binding)| *binding).collect::<Vec<_>>();
        match kind {
            EncodingKind::Ggml => ggml(&bindings, interleaved, first),
            EncodingKind::MxFp4 if interleaved && ocp_mxfp4(&bindings) => {
                Ok(Self::InterleavedMxFp4)
            },
            EncodingKind::MxFp4 => Ok(Self::Block(BlockFormat::MxFp4)),
            EncodingKind::MxFp8 => Ok(Self::Block(BlockFormat::MxFp8)),
            EncodingKind::NvFp4 => Ok(Self::Block(BlockFormat::NvFp4)),
            EncodingKind::Affine if interleaved => Err(BankRefusal::Unread(first)),
            EncodingKind::Affine if bindings.iter().all(|binding| affine_bias(binding)) => {
                Ok(Self::Affine)
            },
            EncodingKind::Affine => Err(BankRefusal::Unbiased(first)),
            EncodingKind::Dense => Ok(Self::Dense),
            EncodingKind::PackedInteger
            | EncodingKind::Awq
            | EncodingKind::Gptq
            | EncodingKind::BitsAndBytes4Bit
            | EncodingKind::Float8 => Err(BankRefusal::Unread(first)),
        }
    }

    /// The layout CUDA loads these banks in; none when it loads them in no
    /// layout. Metal loads every variant, each layer on its own.
    pub(super) const fn cuda_layout(self) -> Option<CudaExpertLayout> {
        match self {
            Self::GgmlMxFp4 => Some(CudaExpertLayout::Ggml),
            Self::InterleavedMxFp4 => Some(CudaExpertLayout::Native),
            Self::Affine => Some(CudaExpertLayout::Mlx),
            Self::Dense => Some(CudaExpertLayout::Dense),
            Self::Block(_) => None,
        }
    }

    /// The encoding family of the banks.
    pub(super) const fn kind(self) -> EncodingKind {
        match self {
            Self::GgmlMxFp4 => EncodingKind::Ggml,
            Self::InterleavedMxFp4 | Self::Block(BlockFormat::MxFp4) => EncodingKind::MxFp4,
            Self::Block(BlockFormat::MxFp8) => EncodingKind::MxFp8,
            Self::Block(BlockFormat::NvFp4) => EncodingKind::NvFp4,
            Self::Affine => EncodingKind::Affine,
            Self::Dense => EncodingKind::Dense,
        }
    }

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::GgmlMxFp4 => "separate GGML MXFP4 banks",
            Self::InterleavedMxFp4 => "interleaved OCP MXFP4 gate/up and down banks",
            Self::Affine => "separate grouped-affine banks",
            Self::Dense => "dense banks",
            Self::Block(BlockFormat::MxFp4) => "MXFP4 banks outside the interleaved OCP layout",
            Self::Block(BlockFormat::MxFp8) => "MXFP8 banks",
            Self::Block(BlockFormat::NvFp4) => "NVFP4 banks",
        }
    }
}

impl BankRefusal {
    /// The encoding family the refusal concerns: for mixed banks, the first
    /// projection stored apart from the first.
    pub(super) fn kind(&self) -> EncodingKind {
        match self {
            Self::Mixed(encodings) => {
                let first = encodings[0].1.kind();
                encodings
                    .iter()
                    .map(|(_, encoding)| encoding.kind())
                    .find(|kind| *kind != first)
                    .unwrap_or(first)
            },
            Self::GgmlType(_) => EncodingKind::Ggml,
            Self::Unbiased(encoding) | Self::Unread(encoding) => encoding.kind(),
            Self::Unstacked(kind) => *kind,
        }
    }
}

fn ggml(
    bindings: &[&TensorBinding],
    interleaved: bool,
    first: WeightEncoding,
) -> Result<ExpertBanks, BankRefusal> {
    for binding in bindings {
        if let TensorStorage::Ggml { format, .. } = &binding.storage
            && format.ggml_type != GgmlType::MXFP4
        {
            return Err(BankRefusal::GgmlType(format.ggml_type));
        }
    }
    if interleaved {
        return Err(BankRefusal::Unread(first));
    }
    let biased = bindings.iter().all(|binding| {
        matches!(&binding.storage, TensorStorage::Ggml { companion: GgmlCompanion::Bias(_), .. })
    });
    if biased {
        Ok(ExpertBanks::GgmlMxFp4)
    } else {
        Err(BankRefusal::Unbiased(first))
    }
}

/// Interleaved OCP MXFP4 gate/up and down banks with biases, the layout
/// CUDA's native loader reads.
fn ocp_mxfp4(bindings: &[&TensorBinding]) -> bool {
    let [gate_up, down] = bindings else {
        return false;
    };
    let biased = bindings.iter().all(|binding| {
        matches!(&binding.storage, TensorStorage::BlockQuantized { bias: Some(_), .. })
    });
    let fused = matches!(
        gate_up.block_projection_layout(),
        Some(BlockProjectionLayout::FusedGateUpBank { interleaved: true, .. })
    );
    let bank =
        matches!(down.block_projection_layout(), Some(BlockProjectionLayout::MatrixBank { .. }));
    biased && fused && bank
}

/// A binding's encoding family; an auxiliary tensor is plain dense storage.
fn family(binding: &TensorBinding) -> EncodingKind {
    WeightEncoding::of(&binding.storage).map_or(EncodingKind::Dense, |encoding| encoding.kind())
}

fn affine_bias(binding: &TensorBinding) -> bool {
    matches!(&binding.storage, TensorStorage::AffineQuantized { output_bias: Some(_), .. })
}
