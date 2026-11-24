//! Fused projections that llama.cpp keeps as one tensor, split back into
//! the checkpoint's projections in the catalog: a row range of a GGML (or
//! dense) matrix is a contiguous byte range of whole rows, so each part reads
//! in place. Phi-3/Phi-4 and GLM-4 store `attn_qkv` (`[q | k | v]` rows,
//! and their bias) and `ffn_up` with the gate and up rows (`[gate | up]`,
//! which `swiglu` splits in that order).

use super::GgufHeader;
use crate::{
    error::{ModelsError, Result},
    weights::TensorInfo,
};

pub(super) fn split(header: &GgufHeader, tensors: &mut Vec<TensorInfo>) -> Result<()> {
    let Some(architecture) = header.architecture().filter(|arch| matches!(*arch, "phi3" | "glm4"))
    else {
        return Ok(());
    };
    let key = |name: &str| header.unsigned(&format!("{architecture}.{name}"));
    let (Some(hidden), Some(heads)) = (key("embedding_length"), key("attention.head_count")) else {
        return Ok(());
    };
    let kv_heads = key("attention.head_count_kv").unwrap_or(heads);
    let head_dim = key("attention.key_length").unwrap_or_else(|| hidden / heads.max(1));
    let width = |heads: u64| usize::try_from(heads * head_dim);
    let (query, key_value) = (width(heads)?, width(kv_heads)?);
    let feed_forward = key("feed_forward_length").map(usize::try_from).transpose()?;
    let mut split = Vec::with_capacity(tensors.len());
    for tensor in tensors.drain(..) {
        let parts: Option<Vec<(&str, usize)>> = if tensor.name.ends_with(".attn_qkv.weight")
            || tensor.name.ends_with(".attn_qkv.bias")
        {
            Some(vec![("attn_q", query), ("attn_k", key_value), ("attn_v", key_value)])
        } else if tensor.name.ends_with(".ffn_up.weight")
            && feed_forward.is_some_and(|ff| tensor.shape.first() == Some(&(2 * ff)))
        {
            let ff = feed_forward.unwrap_or_default();
            Some(vec![("ffn_gate", ff), ("ffn_up", ff)])
        } else {
            None
        };
        match parts {
            Some(parts) => split.extend(rows(&tensor, &parts)?),
            None => split.push(tensor),
        }
    }
    *tensors = split;
    Ok(())
}

/// `tensor` cut into consecutive row ranges named `<block>.<role>.weight`.
fn rows(tensor: &TensorInfo, parts: &[(&str, usize)]) -> Result<Vec<TensorInfo>> {
    let invalid = |reason: &str| ModelsError::InvalidGguf(format!("{}: {reason}", tensor.name));
    let Some((&rows, columns)) = tensor.shape.split_first() else {
        return Err(invalid("a fused projection has no rows"));
    };
    if parts.iter().map(|(_, count)| count).sum::<usize>() != rows {
        return Err(invalid("fused rows do not add up to the projections"));
    }
    let bytes = tensor.payload_bytes()?;
    if rows == 0 || !bytes.is_multiple_of(rows) {
        return Err(invalid("fused rows are not whole"));
    }
    let row_bytes = u64::try_from(bytes / rows)?;
    let (stem, suffix) = tensor.name.rsplit_once('.').ok_or_else(|| invalid("has no suffix"))?;
    let (block, _) = stem.rsplit_once('.').ok_or_else(|| invalid("is not a block tensor"))?;
    let mut start = tensor.data_offsets[0];
    parts
        .iter()
        .map(|&(role, count)| {
            let end = start + row_bytes * u64::try_from(count)?;
            let mut shape = vec![count];
            shape.extend_from_slice(columns);
            let part = TensorInfo {
                name: format!("{block}.{role}.{suffix}"),
                shape,
                data_offsets: [start, end],
                ..tensor.clone()
            };
            start = end;
            Ok(part)
        })
        .collect()
}
