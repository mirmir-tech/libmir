use std::path::PathBuf;

use super::*;
use crate::{
    layout::{PooledVisionConfig, SpatialMergeVisionConfig},
    weights::TensorInfo,
};

#[test]
fn pooled_schema_covers_every_encoder_layer() {
    let schema = VisionTensorSchema::discover(&VisionConfig::PooledEncoder(pooled_config()));
    let readiness = schema.readiness(&catalog_for(&schema, 0));
    assert!(readiness.is_ready());
    assert_eq!(readiness.required, 5 + 2 * (13 + 7 * 4));
}

#[test]
fn pooled_schema_accepts_root_scoped_mlx_names() {
    let schema = VisionTensorSchema::discover(&VisionConfig::PooledEncoder(pooled_config()));
    assert!(schema.readiness(&catalog_for(&schema, 1)).is_ready());
}

#[test]
fn spatial_merge_schema_reports_a_missing_block_tensor() {
    let schema =
        VisionTensorSchema::discover(&VisionConfig::SpatialMergeEncoder(spatial_merge_config()));
    let mut catalog = catalog_for(&schema, 0);
    catalog
        .tensors
        .retain(|tensor| tensor.name != "model.visual.blocks.1.attn.qkv.bias");
    let readiness = schema.readiness(&catalog);
    assert!(!readiness.is_ready());
    assert_eq!(readiness.missing.len(), 1);
    assert!(readiness.missing[0].contains("vision QKV bias"));
}

#[test]
fn spatial_merge_schema_accepts_alternate_vision_tower_prefix() {
    let schema =
        VisionTensorSchema::discover(&VisionConfig::SpatialMergeEncoder(spatial_merge_config()));
    assert!(schema.readiness(&catalog_for(&schema, 1)).is_ready());
}

#[test]
fn readiness_reports_distinct_physical_dtypes() {
    let schema = VisionTensorSchema::discover(&VisionConfig::PooledEncoder(pooled_config()));
    let mut catalog = catalog_for(&schema, 0);
    for tensor in &mut catalog.tensors {
        tensor.dtype = if tensor.name.ends_with("std_scale") {
            "F32"
        } else {
            "F16"
        }
        .into();
    }

    assert_eq!(schema.readiness(&catalog).dtypes, ["F16", "F32"]);
}

#[test]
fn pooled_bound_projection_does_not_claim_its_packed_container_as_dense() {
    let schema = VisionTensorSchema::discover(&VisionConfig::PooledEncoder(pooled_config()));
    let mut catalog = catalog_for(&schema, 1);
    let mut found = false;
    for tensor in &mut catalog.tensors {
        if tensor.name == "embed_vision.embedding_projection.weight" {
            tensor.dtype = "U32".into();
            found = true;
        }
    }

    assert!(found);
    assert_eq!(schema.readiness(&catalog).dtypes, ["BF16"]);
}

fn catalog_for(schema: &VisionTensorSchema, alias: usize) -> TensorCatalog {
    TensorCatalog::new(
        schema
            .requirements
            .iter()
            .filter_map(|requirement| requirement.aliases.get(alias))
            .map(|name| TensorInfo {
                name: name.clone(),
                file: PathBuf::new(),
                dtype: "BF16".into(),
                shape: Vec::new(),
                data_start: 0,
                data_offsets: [0, 0],
            })
            .collect(),
    )
}

fn pooled_config() -> PooledVisionConfig {
    PooledVisionConfig {
        hidden_size: 8,
        output_hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 4,
        patch_size: 2,
        pooling_kernel_size: 2,
        position_embedding_size: 16,
        rms_norm_eps: 1.0e-6,
        rope_theta: 100.0,
        hidden_activation: "gelu_pytorch_tanh".into(),
        use_clipped_linears: true,
        standardize: true,
        image_token_id: 10,
        image_begin_token_id: 11,
        image_end_token_id: 12,
        soft_tokens_per_image: 4,
        bidirectional_image_attention: true,
    }
}

fn spatial_merge_config() -> SpatialMergeVisionConfig {
    SpatialMergeVisionConfig {
        hidden_size: 8,
        output_hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        in_channels: 3,
        patch_size: 2,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        num_position_embeddings: 16,
        hidden_activation: "gelu_pytorch_tanh".into(),
        image_token_id: 10,
        vision_start_token_id: 11,
        vision_end_token_id: 12,
        mrope_interleaved: true,
        mrope_sections: vec![1, 1, 2],
    }
}
