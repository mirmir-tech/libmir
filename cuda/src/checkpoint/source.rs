use models::weights::{TensorCatalog, TensorInfo};

use crate::{Error, NvFp4ExpertSource, Result};

const TENSOR_SUFFIXES: [&str; 20] = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "post_feedforward_layernorm_1.weight",
    "router.proj.weight",
    "router.scale",
    "router.per_expert_scale",
    "pre_feedforward_layernorm_2.weight",
    "post_feedforward_layernorm_2.weight",
    "post_feedforward_layernorm.weight",
    "layer_scalar",
];

pub(super) struct LayerSource<'a> {
    pub prefix: String,
    pub tensors: Vec<&'a TensorInfo>,
    pub names: Vec<String>,
    pub gate: Vec<NvFp4ExpertSource<'a>>,
    pub up: Vec<NvFp4ExpertSource<'a>>,
    pub down: Vec<NvFp4ExpertSource<'a>>,
}

impl<'a> LayerSource<'a> {
    pub fn discover(
        catalog: &'a TensorCatalog,
        layer: usize,
        experts: usize,
        key_is_value: bool,
    ) -> Result<Self> {
        let prefix = layer_prefix(catalog, layer)?;
        let names = TENSOR_SUFFIXES
            .iter()
            .enumerate()
            .map(|(index, suffix)| {
                let suffix = if key_is_value && index == 3 {
                    TENSOR_SUFFIXES[2]
                } else {
                    suffix
                };
                format!("{prefix}.{suffix}")
            })
            .collect::<Vec<_>>();
        let mut tensors = Vec::with_capacity(names.len());
        for name in &names {
            let tensor = required(catalog, name)?;
            if !tensors.iter().any(|present: &&TensorInfo| present.name == tensor.name) {
                tensors.push(tensor);
            }
        }
        Ok(Self {
            gate: expert_sources(catalog, &prefix, experts, "gate_proj")?,
            up: expert_sources(catalog, &prefix, experts, "up_proj")?,
            down: expert_sources(catalog, &prefix, experts, "down_proj")?,
            prefix,
            tensors,
            names,
        })
    }
}

fn layer_prefix(catalog: &TensorCatalog, layer: usize) -> Result<String> {
    [
        format!("model.layers.{layer}"),
        format!("language_model.model.layers.{layer}"),
        format!("model.language_model.layers.{layer}"),
    ]
    .into_iter()
    .find(|prefix| catalog.contains(&format!("{prefix}.input_layernorm.weight")))
    .ok_or_else(|| Error::MissingTensor(format!("decoder layer {layer} input norm")))
}

fn expert_sources<'a>(
    catalog: &'a TensorCatalog,
    prefix: &str,
    experts: usize,
    projection: &str,
) -> Result<Vec<NvFp4ExpertSource<'a>>> {
    (0..experts)
        .map(|expert| {
            let prefix = format!("{prefix}.experts.{expert}.{projection}");
            Ok(NvFp4ExpertSource {
                weight: required(catalog, &format!("{prefix}.weight"))?,
                weight_scale: required(catalog, &format!("{prefix}.weight_scale"))?,
                weight_scale_2: required(catalog, &format!("{prefix}.weight_scale_2"))?,
                input_scale: required(catalog, &format!("{prefix}.input_scale"))?,
            })
        })
        .collect()
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}
