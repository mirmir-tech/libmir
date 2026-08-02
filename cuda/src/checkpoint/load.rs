use mircuda::{DeviceBuffer, bf16};
use models::{
    layout::DecoderConfig,
    semantic::SemanticModelSpec,
    weights::{HybridMoeLayerBindings, TensorCatalog, WeightBindingPlan},
};

use super::{NvFp4MoeLayerLoadConfig, model::payload_bytes, source::LayerSource};
use crate::{
    Bf16LinearPackWeights, Bf16LinearPairWeights, CudaBackend, CudaTensor, CudaTensorSet,
    DecodeAttentionOutputWeight, DecodeAttentionWeights, DecodeMoeBlockWeights,
    DecodeMoeLayerTemplate, DecodeQkvWeights, Error, NvFp4ExpertBankConfig, Result, RouterTensors,
};

impl CudaBackend {
    /// Loads one config-discovered NVFP4 routed-MoE layer into an immutable
    /// model-owned template.
    pub fn load_nvfp4_moe_layer_template(
        &self,
        decoder: &DecoderConfig,
        catalog: &TensorCatalog,
        layer: usize,
        load: NvFp4MoeLayerLoadConfig,
    ) -> Result<DecodeMoeLayerTemplate> {
        let spec = SemanticModelSpec::discover(decoder, catalog)?;
        let bindings = WeightBindingPlan::discover(&spec, catalog)?;
        let layer_bindings = bindings.hybrid_moe_layer(layer)?;
        Ok(self
            .load_nvfp4_moe_layer_template_tracked(decoder, catalog, layer, &layer_bindings, load)?
            .0)
    }

    pub(super) fn load_nvfp4_moe_layer_template_tracked(
        &self,
        decoder: &DecoderConfig,
        catalog: &TensorCatalog,
        layer: usize,
        bindings: &HybridMoeLayerBindings<'_>,
        load: NvFp4MoeLayerLoadConfig,
    ) -> Result<(DecodeMoeLayerTemplate, u64)> {
        let block = load.block(decoder, layer)?;
        let source = LayerSource::discover(catalog, bindings)?;
        let tensors = upload(self, &source)?;
        let gate = self.prepare_nvfp4_expert_bank(
            bank(block.experts, block.attention.hidden_size, block.expert_intermediate),
            &source.gate,
        )?;
        let up = self.prepare_nvfp4_expert_bank(
            bank(block.experts, block.attention.hidden_size, block.expert_intermediate),
            &source.up,
        )?;
        let down = self.prepare_nvfp4_expert_bank(
            bank(block.experts, block.expert_intermediate, block.attention.hidden_size),
            &source.down,
        )?;
        let dense_gate_up = self.pack_bf16_linear_pair(
            tensor(&tensors, &source.names[9])?,
            tensor(&tensors, &source.names[10])?,
        )?;
        let qkv = self.pack_bf16_linears([
            tensor(&tensors, &source.names[1])?,
            tensor(&tensors, &source.names[2])?,
            tensor(&tensors, &source.names[3])?,
        ])?;
        let weights = weights(&tensors, &source.names, &qkv, &dense_gate_up)?;
        tracing::debug!(
            backend = "cuda",
            layer,
            tensors = source.tensors.len(),
            experts = block.experts,
            "loaded NVFP4 routed MoE layer template"
        );
        let bytes = source.payload_bytes()?;
        let template = self.prepare_decode_moe_layer_template(block, weights, gate, up, down)?;
        Ok((template, bytes))
    }
}

impl LayerSource<'_> {
    fn payload_bytes(&self) -> Result<u64> {
        let experts = self.gate.iter().chain(&self.up).chain(&self.down).flat_map(|source| {
            [source.weight, source.weight_scale, source.weight_scale_2, source.input_scale]
        });
        payload_bytes(self.tensors.iter().copied().chain(experts))
    }
}

fn upload(backend: &CudaBackend, source: &LayerSource<'_>) -> Result<CudaTensorSet> {
    let cast = backend.prepare_dense_cast()?;
    let mut upload = backend.begin_tensor_upload();
    for tensor in &source.tensors {
        upload.enqueue_float_as_bf16(tensor, &cast)?;
    }
    upload.finish()
}

fn bank(experts: usize, input: usize, output: usize) -> NvFp4ExpertBankConfig {
    NvFp4ExpertBankConfig {
        experts,
        input_features: input,
        output_features: output,
    }
}

pub(super) fn weights<'a>(
    tensors: &'a CudaTensorSet,
    names: &[String],
    qkv: &'a Bf16LinearPackWeights<3>,
    dense_gate_up: &'a Bf16LinearPairWeights,
) -> Result<DecodeMoeBlockWeights<'a>> {
    let get = |index: usize| tensor(tensors, &names[index]);
    Ok(DecodeMoeBlockWeights {
        attention: DecodeAttentionWeights {
            input_norm: get(0)?,
            qkv: DecodeQkvWeights::Bf16(qkv),
            query_norm: bf16_tensor(get(4)?)?,
            key_norm: bf16_tensor(get(5)?)?,
            output: DecodeAttentionOutputWeight::Bf16(get(6)?),
        },
        post_attention_norm: get(7)?,
        pre_dense_norm: get(8)?,
        dense_gate_up,
        dense_down: get(11)?,
        post_dense_norm: get(12)?,
        router: RouterTensors {
            projection: get(13)?,
            norm_scale: get(14)?,
            expert_scale: get(15)?,
        },
        pre_expert_norm: get(16)?,
        post_expert_norm: get(17)?,
        post_feed_forward_norm: get(18)?,
        layer_scalar: get(19)?,
    })
}

pub(super) fn tensor<'a>(tensors: &'a CudaTensorSet, name: &str) -> Result<&'a CudaTensor> {
    tensors.get(name).ok_or_else(|| Error::MissingTensor(name.into()))
}

fn bf16_tensor(tensor: &CudaTensor) -> Result<&DeviceBuffer<bf16>> {
    tensor.as_bf16().ok_or_else(|| Error::DTypeMismatch {
        name: tensor.name().into(),
        expected: "BF16",
    })
}
