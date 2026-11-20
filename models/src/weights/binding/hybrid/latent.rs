//! Typed bindings of one multi-head latent attention decoder layer.

use super::{
    MixedFeedForwardBindings, attention, invalid, layer, mixed::mixed_feed_forward, optional,
};
use crate::{
    error::Result,
    weights::{
        AttentionProjectionRole, LatentAttentionTensorRole, LayerTensorRole, TensorBinding,
        WeightBindingPlan,
    },
};

#[derive(Debug, Clone, Copy)]
pub struct LatentDecoderLayerBindings<'a> {
    pub input_norm: &'a TensorBinding,
    pub attention: LatentAttentionBindings<'a>,
    pub post_attention_norm: &'a TensorBinding,
    pub feed_forward: MixedFeedForwardBindings<'a>,
}

#[derive(Debug, Clone, Copy)]
pub struct LatentAttentionBindings<'a> {
    pub query: LatentQueryBindings<'a>,
    /// `kv_a_proj_with_mqa`: hidden state to the latent and the rotary key.
    pub key_value_down: &'a TensorBinding,
    pub key_value_norm: &'a TensorBinding,
    /// `kv_b_proj`: latent to every head's position-free key and value.
    pub key_value_up: &'a TensorBinding,
    pub output: &'a TensorBinding,
}

/// How a layer projects its queries.
#[derive(Debug, Clone, Copy)]
pub enum LatentQueryBindings<'a> {
    /// One projection from the hidden state (`q_proj`).
    Direct(&'a TensorBinding),
    /// Down-projection, normalized query latent, up-projection.
    Latent {
        down: &'a TensorBinding,
        norm: &'a TensorBinding,
        up: &'a TensorBinding,
    },
}

impl WeightBindingPlan {
    pub fn latent_decoder_layer(&self, index: usize) -> Result<LatentDecoderLayerBindings<'_>> {
        let get = |tensor| layer(self, index, LayerTensorRole::LatentAttention { tensor });
        let direct = optional(
            self,
            index,
            LayerTensorRole::AttentionProjection {
                projection: AttentionProjectionRole::Query,
            },
        );
        let latent_down = optional(
            self,
            index,
            LayerTensorRole::LatentAttention {
                tensor: LatentAttentionTensorRole::QueryDown,
            },
        );
        let query = match (direct, latent_down) {
            (Some(query), None) => LatentQueryBindings::Direct(query),
            (None, Some(down)) => LatentQueryBindings::Latent {
                down,
                norm: get(LatentAttentionTensorRole::QueryLatentNorm)?,
                up: get(LatentAttentionTensorRole::QueryUp)?,
            },
            _ => return Err(invalid(index, "latent attention needs exactly one query path")),
        };
        Ok(LatentDecoderLayerBindings {
            input_norm: layer(self, index, LayerTensorRole::InputNorm)?,
            attention: LatentAttentionBindings {
                query,
                key_value_down: get(LatentAttentionTensorRole::KeyValueDown)?,
                key_value_norm: get(LatentAttentionTensorRole::KeyValueLatentNorm)?,
                key_value_up: get(LatentAttentionTensorRole::KeyValueUp)?,
                output: attention(self, index, AttentionProjectionRole::Output)?,
            },
            post_attention_norm: layer(self, index, LayerTensorRole::PostAttentionNorm)?,
            feed_forward: mixed_feed_forward(self, index)?,
        })
    }
}

impl<'a> LatentDecoderLayerBindings<'a> {
    #[must_use]
    pub fn physical_sources(self) -> Vec<&'a str> {
        let attention = self.attention;
        let mut bindings = vec![
            self.input_norm,
            self.post_attention_norm,
            attention.key_value_down,
            attention.key_value_norm,
            attention.key_value_up,
            attention.output,
        ];
        match attention.query {
            LatentQueryBindings::Direct(query) => bindings.push(query),
            LatentQueryBindings::Latent { down, norm, up } => bindings.extend([down, norm, up]),
        }
        bindings.extend(self.feed_forward.bindings());
        super::sources(bindings)
    }
}
