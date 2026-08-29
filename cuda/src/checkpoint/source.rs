use std::collections::HashSet;

use models::weights::{
    BlockQuantization, HybridMoeExpertBindings, HybridMoeLayerBindings, TensorBinding,
    TensorCatalog, TensorInfo, TensorStorage,
};

use crate::{Error, NvFp4ExpertSource, NvFp4ScaleMode, Result};

pub(super) struct LayerSource<'a> {
    pub tensors: Vec<&'a TensorInfo>,
    pub names: Vec<String>,
    pub gate: Vec<NvFp4ExpertSource<'a>>,
    pub up: Vec<NvFp4ExpertSource<'a>>,
    pub down: Vec<NvFp4ExpertSource<'a>>,
}

impl<'a> LayerSource<'a> {
    pub fn discover(
        catalog: &'a TensorCatalog,
        bindings: &HybridMoeLayerBindings<'_>,
    ) -> Result<Self> {
        let names = common(bindings).into_iter().map(|binding| binding.source.clone()).collect();
        let mut seen = HashSet::new();
        let tensors = common(bindings)
            .into_iter()
            .flat_map(TensorBinding::physical_sources)
            .filter(|name| seen.insert((*name).to_owned()))
            .map(|name| required(catalog, name))
            .collect::<Result<_>>()?;
        let HybridMoeExpertBindings::Individual { gate, up, down } = &bindings.experts else {
            return Err(Error::UnsupportedDecoderLayer(
                "CUDA hybrid MoE requires individual NVFP4 expert bindings".into(),
            ));
        };
        Ok(Self {
            tensors,
            names,
            gate: expert_sources(catalog, gate)?,
            up: expert_sources(catalog, up)?,
            down: expert_sources(catalog, down)?,
        })
    }
}

pub(super) fn common<'a>(bindings: &'a HybridMoeLayerBindings<'a>) -> Vec<&'a TensorBinding> {
    vec![
        bindings.input_norm,
        bindings.attention.query,
        bindings.attention.key,
        bindings.attention.value.unwrap_or(bindings.attention.key),
        bindings.attention.query_norm,
        bindings.attention.key_norm,
        bindings.attention.output,
        bindings.post_attention_norm,
        bindings.pre_dense_norm,
        bindings.dense.gate,
        bindings.dense.up,
        bindings.dense.down,
        bindings.post_dense_norm,
        bindings.router.projection,
        bindings.router.norm_scale,
        bindings.router.expert_scale,
        bindings.pre_expert_norm,
        bindings.post_expert_norm,
        bindings.post_feed_forward_norm,
        bindings.layer_scale,
    ]
}

fn expert_sources<'a>(
    catalog: &'a TensorCatalog,
    bindings: &[&TensorBinding],
) -> Result<Vec<NvFp4ExpertSource<'a>>> {
    bindings.iter().map(|binding| expert_source(catalog, binding)).collect()
}

fn expert_source<'a>(
    catalog: &'a TensorCatalog,
    binding: &TensorBinding,
) -> Result<NvFp4ExpertSource<'a>> {
    let TensorStorage::BlockQuantized {
        format: BlockQuantization::NVFP4,
        scales,
        global_scale: Some(global_scale),
        input_scale: Some(input_scale),
        ..
    } = &binding.storage
    else {
        return Err(Error::UnsupportedDecoderLayer(format!(
            "CUDA expert {} requires a complete NVFP4 binding",
            binding.source
        )));
    };
    Ok(NvFp4ExpertSource {
        weight: required(catalog, &binding.source)?,
        weight_scale: required(catalog, scales)?,
        weight_scale_2: required(catalog, global_scale)?,
        input_scale: required(catalog, input_scale)?,
        scale_mode: NvFp4ScaleMode::from_names(global_scale, input_scale)?,
    })
}

fn required<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}
