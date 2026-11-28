//! GGML block-quantized weights read in place from GGUF files.
//!
//! A GGML block carries its own scales, so the one companion is a dense bias
//! or an F32 output scale (llama.cpp's `.scale`): the type alone fixes block
//! size, byte layout and dequantization.
//! The logical shape is `[rows, input]` and every row holds whole blocks.

use std::collections::BTreeSet;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{TensorBinding, TensorStorage};
use crate::{
    error::{ModelsError, Result},
    weights::{DenseDType, GgmlType, TensorCatalog, TensorInfo},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GgmlBlockQuantization {
    #[serde(serialize_with = "type_name", deserialize_with = "parse_type")]
    pub ggml_type: GgmlType,
}

/// What a GGML block weight carries beside its blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgmlCompanion {
    /// The blocks alone.
    Plain,
    /// A dense output bias (gpt-oss).
    Bias(String),
    /// An F32 output scale: one value, or one per expert of a bank
    /// (llama.cpp's `.scale`, e.g. `ModelOpt` NVFP4's `weight_scale_2`).
    Scale(String),
}

impl GgmlCompanion {
    #[must_use]
    pub fn bias(&self) -> Option<&str> {
        match self {
            Self::Bias(bias) => Some(bias),
            Self::Plain | Self::Scale(_) => None,
        }
    }

    #[must_use]
    pub fn scale(&self) -> Option<&str> {
        match self {
            Self::Scale(scale) => Some(scale),
            Self::Plain | Self::Bias(_) => None,
        }
    }
}

/// Block storage for a tensor whose catalog dtype names a quantized GGML
/// type, with its companion if the catalog has one; dense GGML types (F32,
/// F16, BF16) stay dense. No runtime applies both a bias and an output scale,
/// so a weight with both is refused.
pub(super) fn storage(
    tensor: &TensorInfo,
    catalog: &TensorCatalog,
    consumed: &mut BTreeSet<String>,
) -> Result<Option<TensorStorage>> {
    let Some(ggml_type) = tensor.dtype.as_ggml() else {
        return Ok(None);
    };
    let stem = tensor.name.trim_end_matches(".weight");
    let mut companion = |suffix: &str| {
        let name = format!("{stem}.{suffix}");
        catalog.contains(&name).then(|| {
            let _inserted = consumed.insert(name.clone());
            name
        })
    };
    // The activation scale of an FP8 or NVFP4 export: llama.cpp loads it but
    // computes with the activations as they are, and so does this runtime.
    let _input_scale = companion("input_scale");
    let companion = match (companion("bias"), companion("scale")) {
        (None, None) => GgmlCompanion::Plain,
        (Some(bias), None) => GgmlCompanion::Bias(bias),
        (None, Some(scale)) => GgmlCompanion::Scale(scale),
        (Some(_), Some(_)) => {
            return Err(ModelsError::InvalidConfig(format!(
                "{} carries both a bias and an output scale",
                tensor.name
            )));
        },
    };
    Ok(Some(TensorStorage::Ggml {
        format: GgmlBlockQuantization { ggml_type },
        companion,
    }))
}

/// Checks the logical shape against the type's blocks and the payload size.
pub(super) fn validate(
    binding: &TensorBinding,
    logical: &[usize],
    format: GgmlBlockQuantization,
    catalog: &TensorCatalog,
) -> Result<()> {
    let invalid = |detail: String| {
        ModelsError::InvalidConfig(format!("{} ({}): {detail}", binding.source, format.ggml_type))
    };
    if binding.shape != logical {
        return Err(invalid(format!(
            "shape {:?} differs from the logical {logical:?}",
            binding.shape
        )));
    }
    let (&input, rows) = logical.split_last().ok_or_else(|| invalid("has no dimensions".into()))?;
    let row_bytes = format.ggml_type.row_bytes(u64::try_from(input)?)?;
    let rows: u64 = rows
        .iter()
        .map(|&rows| u64::try_from(rows))
        .product::<std::result::Result<u64, _>>()?;
    let tensor = catalog
        .get(&binding.source)
        .ok_or_else(|| invalid("is not in the catalog".into()))?;
    let payload = u64::try_from(tensor.payload_bytes()?)?;
    if Some(payload) != rows.checked_mul(row_bytes) {
        return Err(invalid(format!("holds {payload} bytes; {rows} rows of {row_bytes} expected")));
    }
    if let TensorStorage::Ggml {
        companion: GgmlCompanion::Scale(scale), ..
    } = &binding.storage
    {
        // One scale per matrix: a bank `[experts, rows, input]` has one per
        // expert.
        let expected = if logical.len() == 3 {
            logical[0]
        } else {
            1
        };
        let tensor = catalog.get(scale).ok_or_else(|| invalid(format!("{scale} is missing")))?;
        if tensor.dtype != DenseDType::F32 || tensor.shape.iter().product::<usize>() != expected {
            return Err(invalid(format!(
                "scale {scale} is {} {:?}; {expected} F32 values expected",
                tensor.dtype, tensor.shape
            )));
        }
    }
    Ok(())
}

#[expect(clippy::trivially_copy_pass_by_ref, reason = "serde `serialize_with` passes a reference")]
fn type_name<S: Serializer>(
    ggml_type: &GgmlType,
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    serializer.serialize_str(ggml_type.name())
}

fn parse_type<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<GgmlType, D::Error> {
    let name = String::deserialize(deserializer)?;
    GgmlType::from_name(&name)
        .ok_or_else(|| serde::de::Error::custom(format!("unknown GGML type {name}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_ggml_types_bind_with_their_dense_bias() -> Result<()> {
        let info = |name: &str, dtype: &str| TensorInfo {
            name: name.into(),
            file: "model.gguf".into(),
            container: crate::layout::WeightContainer::Gguf,
            dtype: crate::weights::TensorDType::parse(dtype),
            shape: vec![8, 256],
            data_start: 0,
            data_offsets: [0, 8 * 144],
        };
        let weight = "model.layers.0.mlp.up_proj.weight";
        let bias = "model.layers.0.mlp.up_proj.bias";
        let bare = TensorCatalog::new(vec![info(weight, "Q4_K")]);
        let biased = TensorCatalog::new(vec![info(weight, "Q4_K"), info(bias, "F32")]);
        let mut consumed = BTreeSet::new();
        let storage = storage(&info(weight, "Q4_K"), &bare, &mut consumed)?;
        assert!(matches!(
            storage,
            Some(TensorStorage::Ggml {
                format: GgmlBlockQuantization { ggml_type: GgmlType::Q4_K },
                companion: GgmlCompanion::Plain,
            })
        ));
        assert!(storage.is_some_and(|storage| storage_round_trips(&storage)));
        assert!(matches!(
            super::storage(&info(weight, "Q4_K"), &biased, &mut consumed)?,
            Some(TensorStorage::Ggml { companion: GgmlCompanion::Bias(name), .. }) if name == bias
        ));
        assert!(consumed.contains(bias));
        // An output scale binds; an activation scale is consumed unused.
        let scale = "model.layers.0.mlp.up_proj.scale";
        let input_scale = "model.layers.0.mlp.up_proj.input_scale";
        let scaled = TensorCatalog::new(vec![
            info(weight, "NVFP4"),
            info(scale, "F32"),
            info(input_scale, "F32"),
        ]);
        let storage = super::storage(&info(weight, "NVFP4"), &scaled, &mut consumed)?;
        assert!(matches!(
            &storage,
            Some(TensorStorage::Ggml { companion: GgmlCompanion::Scale(name), .. }) if name == scale
        ));
        assert!(storage.is_some_and(|storage| storage_round_trips(&storage)));
        assert!(consumed.contains(scale) && consumed.contains(input_scale));
        let both =
            TensorCatalog::new(vec![info(weight, "Q4_K"), info(bias, "F32"), info(scale, "F32")]);
        assert!(super::storage(&info(weight, "Q4_K"), &both, &mut consumed).is_err());
        assert_eq!(super::storage(&info(weight, "BF16"), &bare, &mut consumed)?, None);
        assert_eq!(super::storage(&info(weight, "U8"), &bare, &mut consumed)?, None);
        Ok(())
    }

    fn storage_round_trips(storage: &TensorStorage) -> bool {
        serde_json::to_string(storage)
            .ok()
            .and_then(|json| serde_json::from_str::<TensorStorage>(&json).ok())
            .is_some_and(|back| &back == storage)
    }
}
