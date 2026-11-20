use std::collections::HashSet;

use models::weights::{
    DenseDecoderLayerBindings, DenseLayerFeedForwardBindings, TensorBinding, TensorCatalog,
    TensorInfo, TensorStorage,
};

use crate::{CudaBackend, CudaTensorSet, Error, ProjectionFormat, Result};

pub(super) struct DenseLayerSource<'a> {
    pub tensors: Vec<&'a TensorInfo>,
}

impl<'a> DenseLayerSource<'a> {
    pub fn discover(
        catalog: &'a TensorCatalog,
        bindings: DenseDecoderLayerBindings<'_>,
    ) -> Result<Self> {
        let mut seen = HashSet::new();
        let tensors = bindings
            .physical_sources()
            .into_iter()
            .filter(|name| seen.insert(*name))
            .map(|name| required_tensor(catalog, name))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { tensors })
    }

    pub(super) fn upload(
        &self,
        backend: &CudaBackend,
        bindings: DenseDecoderLayerBindings<'_>,
        format: ProjectionFormat,
    ) -> Result<CudaTensorSet> {
        let mut upload = backend.begin_tensor_upload();
        if format == ProjectionFormat::Bf16 {
            let cast = backend.prepare_dense_cast()?;
            for tensor in &self.tensors {
                upload.enqueue_as_bf16(tensor, &cast)?;
            }
        } else if format == ProjectionFormat::Ggml {
            // GGUF keeps norms in F32 beside the GGML blocks.
            let cast = backend.prepare_dense_cast()?;
            for tensor in &self.tensors {
                upload.enqueue_float_as_bf16(tensor, &cast)?;
            }
        } else if format == ProjectionFormat::PackedInteger {
            let metadata = packed_shape_sources(bindings);
            let dense = dense_sources(bindings);
            let cast = backend.prepare_dense_cast()?;
            for tensor in &self.tensors {
                if metadata.contains(tensor.name.as_str()) {
                    continue;
                }
                if dense.contains(tensor.name.as_str()) {
                    upload.enqueue_as_bf16(tensor, &cast)?;
                } else {
                    upload.enqueue(tensor)?;
                }
            }
        } else {
            for tensor in &self.tensors {
                upload.enqueue(tensor)?;
            }
        }
        upload.finish()
    }
}

fn dense_sources(bindings: DenseDecoderLayerBindings<'_>) -> HashSet<&str> {
    layer_bindings(bindings)
        .into_iter()
        .filter(|binding| matches!(binding.storage, TensorStorage::Dense { .. }))
        .flat_map(|binding| binding.physical_sources())
        .collect()
}

fn layer_bindings(bindings: DenseDecoderLayerBindings<'_>) -> Vec<&TensorBinding> {
    let mut values = vec![
        bindings.input_norm,
        bindings.attention.query,
        bindings.attention.key,
        bindings.attention.value,
        bindings.attention.output,
        bindings.post_attention_norm,
    ];
    values.extend(gated(bindings));
    values.extend(bindings.attention.query_norm);
    values.extend(bindings.attention.key_norm);
    if let Some(sandwich) = bindings.sandwich {
        values.extend([sandwich.attention_output_norm, sandwich.feed_forward_output_norm]);
        values.extend(sandwich.layer_scale);
    }
    values
}

fn packed_shape_sources(bindings: DenseDecoderLayerBindings<'_>) -> HashSet<&str> {
    [
        bindings.attention.query,
        bindings.attention.key,
        bindings.attention.value,
        bindings.attention.output,
    ]
    .into_iter()
    .chain(gated(bindings))
    .filter_map(|binding| match &binding.storage {
        TensorStorage::PackedInteger { shape, .. } => Some(shape.as_str()),
        TensorStorage::BitsAndBytes4Bit { quant_state, .. } => Some(quant_state.as_str()),
        _ => None,
    })
    .collect()
}

/// The gated MLP projections; routed layers have none.
fn gated(bindings: DenseDecoderLayerBindings<'_>) -> Vec<&TensorBinding> {
    match bindings.feed_forward {
        DenseLayerFeedForwardBindings::Gated(value) => vec![value.gate, value.up, value.down],
        DenseLayerFeedForwardBindings::Routed(_) => Vec::new(),
    }
}

fn required_tensor<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}
