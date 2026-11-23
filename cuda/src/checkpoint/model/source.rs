//! The decoder boundary a model template loads: embedding, final norm and
//! output head, and the tensors they read.

use models::weights::{TensorBinding, TensorCatalog, TensorInfo, TensorStorage, WeightBindingPlan};

use super::payload_bytes;
use crate::{CudaBackend, CudaTensorSet, Error, Result};

pub(in crate::checkpoint) struct ModelSource<'a> {
    pub(in crate::checkpoint) embedding: &'a TensorBinding,
    pub(in crate::checkpoint) final_norm: &'a str,
    pub(in crate::checkpoint) final_norm_bias: Option<&'a str>,
    pub(in crate::checkpoint) output: &'a TensorBinding,
    tensors: Vec<&'a TensorInfo>,
}

impl<'a> ModelSource<'a> {
    /// The boundary of a runtime that applies no final norm or output head
    /// bias, which it refuses.
    pub(in crate::checkpoint) fn discover(
        tied: bool,
        bindings: &'a WeightBindingPlan,
        catalog: &'a TensorCatalog,
    ) -> Result<Self> {
        let source = Self::discover_biased(tied, bindings, catalog)?;
        if source.final_norm_bias.is_some() || source.output_bias() {
            return Err(Error::UnsupportedDecoderLayer(
                "this CUDA runtime applies no final norm or output head bias".into(),
            ));
        }
        Ok(source)
    }

    /// The boundary with its final norm and output head biases (Phi-3.5-MoE).
    pub(in crate::checkpoint) fn discover_biased(
        tied: bool,
        bindings: &'a WeightBindingPlan,
        catalog: &'a TensorCatalog,
    ) -> Result<Self> {
        let boundary = bindings.decoder_boundary_with_tied_output(tied)?;
        let final_norm_bias = boundary.final_norm.dense_bias();
        let final_norm = required(catalog, &boundary.final_norm.source)?;
        let mut seen = std::collections::HashSet::new();
        let mut tensors = Vec::new();
        for name in boundary
            .embedding
            .physical_sources()
            .into_iter()
            .chain(boundary.output.physical_sources())
            .chain(std::iter::once(boundary.final_norm.source.as_str()))
            .chain(final_norm_bias)
        {
            if seen.insert(name) {
                tensors.push(required(catalog, name)?);
            }
        }
        Ok(Self {
            embedding: boundary.embedding,
            final_norm: &final_norm.name,
            final_norm_bias,
            output: boundary.output,
            tensors,
        })
    }

    fn output_bias(&self) -> bool {
        matches!(
            &self.output.storage,
            TensorStorage::Dense { bias: Some(_), .. } | TensorStorage::Ggml { bias: Some(_), .. }
        )
    }

    pub(in crate::checkpoint) fn upload(&self, backend: &CudaBackend) -> Result<CudaTensorSet> {
        let cast = backend.prepare_dense_cast()?;
        let mut upload = backend.begin_tensor_upload();
        let bindings = [self.embedding, self.output];
        let raw = bindings
            .iter()
            .flat_map(|binding| runtime_raw_sources(binding))
            .collect::<std::collections::HashSet<_>>();
        let metadata = bindings
            .iter()
            .filter_map(|binding| packed_shape_source(binding))
            .collect::<std::collections::HashSet<_>>();
        for tensor in &self.tensors {
            if metadata.contains(tensor.name.as_str()) {
                continue;
            }
            if raw.contains(tensor.name.as_str()) {
                upload.enqueue(tensor)?;
            } else {
                upload.enqueue_as_bf16(tensor, &cast)?;
            }
        }
        upload.finish()
    }

    pub(in crate::checkpoint) fn payload_bytes(&self) -> Result<u64> {
        payload_bytes(self.tensors.iter().copied())
    }
}
pub(super) fn runtime_raw_sources(binding: &TensorBinding) -> Vec<&str> {
    match &binding.storage {
        // The bias epilogue adds a BF16 bias.
        TensorStorage::Ggml { scale, .. } => {
            std::iter::once(binding.source.as_str()).chain(scale.as_deref()).collect()
        },
        TensorStorage::AffineQuantized { .. }
        | TensorStorage::BlockQuantized { .. }
        | TensorStorage::Float8 { .. } => binding.physical_sources(),
        TensorStorage::PackedInteger { scales, .. } => {
            vec![binding.source.as_str(), scales.as_str()]
        },
        _ => Vec::new(),
    }
}

pub(super) fn packed_shape_source(binding: &TensorBinding) -> Option<&str> {
    match &binding.storage {
        TensorStorage::PackedInteger { shape, .. } => Some(shape),
        _ => None,
    }
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}
