use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum LogicalTensorRole {
    Embedding,
    FinalNorm,
    Output,
    Layer { index: usize, tensor: LayerTensorRole },
    Auxiliary { path: String },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LayerTensorRole {
    InputNorm,
    PostAttentionNorm,
    PreDenseNorm,
    PostDenseNorm,
    PreExpertNorm,
    PostExpertNorm,
    PostFeedForwardNorm,
    AttentionProjection {
        projection: AttentionProjectionRole,
    },
    QueryNorm,
    KeyNorm,
    AttentionSinks,
    LinearAttention {
        tensor: LinearAttentionTensorRole,
    },
    Router,
    RouterNormScale,
    RouterExpertScale,
    LayerScale,
    FeedForwardProjection {
        projection: FeedForwardProjectionRole,
    },
    ExpertProjection {
        expert: Option<usize>,
        projection: ExpertProjectionRole,
    },
    SharedExpertProjection {
        projection: FeedForwardProjectionRole,
    },
    SharedExpertOutputGate,
    Auxiliary {
        path: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionProjectionRole {
    Query,
    Key,
    Value,
    Qkv,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedForwardProjectionRole {
    Gate,
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertProjectionRole {
    Gate,
    Up,
    GateUp,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinearAttentionTensorRole {
    DecayLog,
    Convolution,
    TimeBias,
    QkvProjection,
    GateProjection,
    AlphaProjection,
    BetaProjection,
    Norm,
    OutputProjection,
}

pub(super) fn parse(name: &str) -> LogicalTensorRole {
    let path = canonical_path(name);
    match path.as_str() {
        "embed_tokens.weight" => return LogicalTensorRole::Embedding,
        "norm.weight" => return LogicalTensorRole::FinalNorm,
        "lm_head.weight" => return LogicalTensorRole::Output,
        _ => {},
    }
    let Some(remainder) = path.strip_prefix("layers.") else {
        return LogicalTensorRole::Auxiliary { path };
    };
    let Some((index, suffix)) = remainder.split_once('.') else {
        return LogicalTensorRole::Auxiliary { path };
    };
    let Ok(index) = index.parse() else {
        return LogicalTensorRole::Auxiliary { path };
    };
    LogicalTensorRole::Layer { index, tensor: layer(suffix) }
}

fn canonical_path(name: &str) -> String {
    let path = name
        .strip_prefix("language_model.model.")
        .or_else(|| name.strip_prefix("model.language_model."))
        .or_else(|| name.strip_prefix("language_model."))
        .or_else(|| name.strip_prefix("model."))
        .unwrap_or(name);
    path.strip_suffix("_blocks")
        .or_else(|| path.strip_suffix(".weight_packed"))
        .map_or_else(|| path.to_owned(), |prefix| format!("{prefix}.weight"))
}

fn layer(suffix: &str) -> LayerTensorRole {
    match suffix {
        "input_layernorm.weight" => LayerTensorRole::InputNorm,
        "post_attention_layernorm.weight" => LayerTensorRole::PostAttentionNorm,
        "pre_feedforward_layernorm.weight" => LayerTensorRole::PreDenseNorm,
        "post_feedforward_layernorm_1.weight" => LayerTensorRole::PostDenseNorm,
        "pre_feedforward_layernorm_2.weight" => LayerTensorRole::PreExpertNorm,
        "post_feedforward_layernorm_2.weight" => LayerTensorRole::PostExpertNorm,
        "post_feedforward_layernorm.weight" => LayerTensorRole::PostFeedForwardNorm,
        "layer_scalar" => LayerTensorRole::LayerScale,
        "self_attn.q_proj.weight" => attention(AttentionProjectionRole::Query),
        "self_attn.k_proj.weight" => attention(AttentionProjectionRole::Key),
        "self_attn.v_proj.weight" => attention(AttentionProjectionRole::Value),
        "self_attn.qkv_proj.weight" | "self_attn.query_key_value.weight" => {
            attention(AttentionProjectionRole::Qkv)
        },
        "self_attn.o_proj.weight" => attention(AttentionProjectionRole::Output),
        "self_attn.q_norm.weight" => LayerTensorRole::QueryNorm,
        "self_attn.k_norm.weight" => LayerTensorRole::KeyNorm,
        "self_attn.sinks" => LayerTensorRole::AttentionSinks,
        "linear_attn.A_log" => linear(LinearAttentionTensorRole::DecayLog),
        "linear_attn.conv1d.weight" => linear(LinearAttentionTensorRole::Convolution),
        "linear_attn.dt_bias" => linear(LinearAttentionTensorRole::TimeBias),
        "linear_attn.in_proj_qkv.weight" => linear(LinearAttentionTensorRole::QkvProjection),
        "linear_attn.in_proj_z.weight" => linear(LinearAttentionTensorRole::GateProjection),
        "linear_attn.in_proj_a.weight" => linear(LinearAttentionTensorRole::AlphaProjection),
        "linear_attn.in_proj_b.weight" => linear(LinearAttentionTensorRole::BetaProjection),
        "linear_attn.norm.weight" => linear(LinearAttentionTensorRole::Norm),
        "linear_attn.out_proj.weight" => linear(LinearAttentionTensorRole::OutputProjection),
        "mlp.router.weight" | "mlp.gate.weight" | "router.proj.weight" => LayerTensorRole::Router,
        "router.scale" => LayerTensorRole::RouterNormScale,
        "router.per_expert_scale" => LayerTensorRole::RouterExpertScale,
        "mlp.gate_proj.weight" => feed_forward(FeedForwardProjectionRole::Gate),
        "mlp.up_proj.weight" => feed_forward(FeedForwardProjectionRole::Up),
        "mlp.down_proj.weight" => feed_forward(FeedForwardProjectionRole::Down),
        "mlp.experts.gate_proj.weight" | "mlp.switch_mlp.gate_proj.weight" => {
            expert(None, ExpertProjectionRole::Gate)
        },
        "mlp.experts.up_proj.weight" | "mlp.switch_mlp.up_proj.weight" => {
            expert(None, ExpertProjectionRole::Up)
        },
        "mlp.experts.gate_up_proj.weight" | "mlp.switch_mlp.gate_up_proj.weight" => {
            expert(None, ExpertProjectionRole::GateUp)
        },
        "mlp.experts.down_proj.weight" | "mlp.switch_mlp.down_proj.weight" => {
            expert(None, ExpertProjectionRole::Down)
        },
        "experts.switch_glu.gate_proj.weight" => expert(None, ExpertProjectionRole::Gate),
        "experts.switch_glu.up_proj.weight" => expert(None, ExpertProjectionRole::Up),
        "experts.switch_glu.down_proj.weight" => expert(None, ExpertProjectionRole::Down),
        "mlp.shared_expert.gate_proj.weight" => shared(FeedForwardProjectionRole::Gate),
        "mlp.shared_expert.up_proj.weight" => shared(FeedForwardProjectionRole::Up),
        "mlp.shared_expert.down_proj.weight" => shared(FeedForwardProjectionRole::Down),
        "mlp.shared_expert_gate.weight" => LayerTensorRole::SharedExpertOutputGate,
        _ => individual_expert(suffix)
            .unwrap_or_else(|| LayerTensorRole::Auxiliary { path: suffix.to_owned() }),
    }
}

const fn attention(projection: AttentionProjectionRole) -> LayerTensorRole {
    LayerTensorRole::AttentionProjection { projection }
}

const fn feed_forward(projection: FeedForwardProjectionRole) -> LayerTensorRole {
    LayerTensorRole::FeedForwardProjection { projection }
}

const fn linear(tensor: LinearAttentionTensorRole) -> LayerTensorRole {
    LayerTensorRole::LinearAttention { tensor }
}

const fn shared(projection: FeedForwardProjectionRole) -> LayerTensorRole {
    LayerTensorRole::SharedExpertProjection { projection }
}

const fn expert(expert: Option<usize>, projection: ExpertProjectionRole) -> LayerTensorRole {
    LayerTensorRole::ExpertProjection { expert, projection }
}

fn individual_expert(suffix: &str) -> Option<LayerTensorRole> {
    let remainder = suffix
        .strip_prefix("experts.")
        .or_else(|| suffix.strip_prefix("mlp.experts."))?;
    let (expert_index, projection) = remainder.split_once('.')?;
    let expert_index = expert_index.parse().ok()?;
    let projection = match projection {
        "gate_proj.weight" => ExpertProjectionRole::Gate,
        "up_proj.weight" => ExpertProjectionRole::Up,
        "down_proj.weight" => ExpertProjectionRole::Down,
        _ => return None,
    };
    Some(expert(Some(expert_index), projection))
}
