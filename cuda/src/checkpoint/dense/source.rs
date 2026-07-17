use models::weights::{TensorCatalog, TensorInfo};

use crate::{Error, ProjectionFormat, Result, kernels::QkvNormalization};

const REQUIRED_SUFFIXES: [&str; 8] = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
];

pub(super) struct DenseLayerSource<'a> {
    pub prefix: String,
    pub tensors: Vec<&'a TensorInfo>,
    pub names: DenseLayerNames,
}

pub(super) struct DenseLayerNames {
    pub required: [String; 8],
    pub query_norm: Option<String>,
    pub key_norm: Option<String>,
    pub down: String,
}

impl<'a> DenseLayerSource<'a> {
    pub fn discover(
        catalog: &'a TensorCatalog,
        layer: usize,
        normalization: QkvNormalization,
        format: ProjectionFormat,
    ) -> Result<Self> {
        let prefix = layer_prefix(catalog, layer)?;
        let required = REQUIRED_SUFFIXES.map(|suffix| format!("{prefix}.{suffix}"));
        let query_norm = optional_norm(catalog, &prefix, "q_norm", normalization.query)?;
        let key_norm = optional_norm(catalog, &prefix, "k_norm", normalization.key)?;
        let down = format!("{prefix}.mlp.down_proj.weight");
        let mut tensors = required
            .iter()
            .map(|name| required_tensor(catalog, name))
            .collect::<Result<Vec<_>>>()?;
        for name in query_norm.iter().chain(&key_norm).chain([&down]) {
            tensors.push(required_tensor(catalog, name)?);
        }
        if format == ProjectionFormat::NvFp4 {
            for name in [
                &required[1], &required[2], &required[3], &required[4], &required[6], &required[7],
                &down,
            ] {
                for auxiliary in nvfp4_auxiliary_names(name)? {
                    tensors.push(required_tensor(catalog, &auxiliary)?);
                }
            }
        }
        Ok(Self {
            prefix,
            tensors,
            names: DenseLayerNames { required, query_norm, key_norm, down },
        })
    }
}

pub(super) fn nvfp4_auxiliary_names(weight: &str) -> Result<[String; 3]> {
    let base = weight
        .strip_suffix(".weight")
        .ok_or_else(|| Error::MissingTensor(format!("invalid projection weight name {weight}")))?;
    Ok([
        format!("{base}.weight_scale"),
        format!("{base}.weight_scale_2"),
        format!("{base}.input_scale"),
    ])
}

fn optional_norm(
    catalog: &TensorCatalog,
    prefix: &str,
    projection: &str,
    required: bool,
) -> Result<Option<String>> {
    let name = format!("{prefix}.self_attn.{projection}.weight");
    if catalog.contains(&name) {
        Ok(Some(name))
    } else if required {
        Err(Error::MissingTensor(name))
    } else {
        Ok(None)
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

fn required_tensor<'a>(catalog: &'a TensorCatalog, name: &str) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| Error::MissingTensor(name.into()))
}
