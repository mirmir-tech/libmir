use super::{
    AttentionProjectionRole, ExpertProjectionRole, FeedForwardProjectionRole, LatentProjectionRole,
    LayerTensorRole, LinearAttentionTensorRole, LogicalTensorRole, StateSpaceTensorRole,
};
use crate::semantic::DecoderSpec;

/// Maps a checkpoint tensor name to its logical role. Paired block layouts
/// translate checkpoint block indices to semantic layer indices.
pub(in crate::weights::binding) fn parse(name: &str, decoder: &DecoderSpec) -> LogicalTensorRole {
    if let Some(path) = backbone_path(name) {
        return super::paired::parse(&path, decoder);
    }
    let path = canonical_path(name);
    match path.as_str() {
        "embed_tokens.weight" => return LogicalTensorRole::Embedding,
        "norm.weight" => return LogicalTensorRole::FinalNorm,
        "lm_head.weight" => return LogicalTensorRole::Output,
        "embed_vision.embedding_projection.weight" => {
            return LogicalTensorRole::VisionProjection;
        },
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

/// Strips the `backbone.` namespace used by state-space hybrid checkpoints.
fn backbone_path(name: &str) -> Option<String> {
    let path = name
        .strip_prefix("language_model.backbone.")
        .or_else(|| name.strip_prefix("backbone."))?;
    Some(path.to_owned())
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
        .or_else(|| path.strip_suffix(".qweight"))
        .map_or_else(|| path.to_owned(), |prefix| format!("{prefix}.weight"))
}

pub(super) fn layer(suffix: &str) -> LayerTensorRole {
    match suffix {
        "input_layernorm.weight" => LayerTensorRole::InputNorm,
        "post_attention_layernorm.weight" => LayerTensorRole::PostAttentionNorm,
        "pre_feedforward_layernorm.weight" => LayerTensorRole::PreDenseNorm,
        "post_feedforward_layernorm_1.weight" => LayerTensorRole::PostDenseNorm,
        "pre_feedforward_layernorm_2.weight" => LayerTensorRole::PreExpertNorm,
        "post_feedforward_layernorm_2.weight" => LayerTensorRole::PostExpertNorm,
        "post_feedforward_layernorm.weight" => LayerTensorRole::PostFeedForwardNorm,
        "layer_scalar" => LayerTensorRole::LayerScale,
        "self_attn.q_proj.weight" | "mixer.q_proj.weight" => {
            attention(AttentionProjectionRole::Query)
        },
        "self_attn.k_proj.weight" | "mixer.k_proj.weight" => {
            attention(AttentionProjectionRole::Key)
        },
        "self_attn.v_proj.weight" | "mixer.v_proj.weight" => {
            attention(AttentionProjectionRole::Value)
        },
        "self_attn.qkv_proj.weight" | "self_attn.query_key_value.weight" => {
            attention(AttentionProjectionRole::Qkv)
        },
        "self_attn.o_proj.weight" | "mixer.o_proj.weight" => {
            attention(AttentionProjectionRole::Output)
        },
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
        "mlp.router.weight" | "mlp.gate.weight" | "router.proj.weight" | "mixer.gate.weight" => {
            LayerTensorRole::Router
        },
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
        "mlp.experts.gate_up_proj.weight"
        | "mlp.experts.gate_up_proj"
        | "mlp.switch_mlp.gate_up_proj.weight"
        | "experts.gate_up_proj" => expert(None, ExpertProjectionRole::GateUp),
        "mlp.experts.down_proj.weight"
        | "mlp.experts.down_proj"
        | "mlp.switch_mlp.down_proj.weight"
        | "experts.down_proj"
        | "experts.switch_glu.down_proj.weight" => expert(None, ExpertProjectionRole::Down),
        "experts.switch_glu.gate_proj.weight" => expert(None, ExpertProjectionRole::Gate),
        "experts.switch_glu.up_proj.weight" => expert(None, ExpertProjectionRole::Up),
        "mlp.shared_expert.gate_proj.weight" => shared(FeedForwardProjectionRole::Gate),
        "mlp.shared_expert.up_proj.weight" | "mixer.shared_experts.up_proj.weight" => {
            shared(FeedForwardProjectionRole::Up)
        },
        "mlp.shared_expert.down_proj.weight" | "mixer.shared_experts.down_proj.weight" => {
            shared(FeedForwardProjectionRole::Down)
        },
        "mlp.shared_expert_gate.weight" => LayerTensorRole::SharedExpertOutputGate,
        "mixer.A_log" => state_space(StateSpaceTensorRole::DecayLog),
        "mixer.D" => state_space(StateSpaceTensorRole::SkipScale),
        "mixer.conv1d.weight" => state_space(StateSpaceTensorRole::Convolution),
        "mixer.dt_bias" => state_space(StateSpaceTensorRole::TimeBias),
        "mixer.in_proj.weight" => state_space(StateSpaceTensorRole::InputProjection),
        "mixer.norm.weight" => state_space(StateSpaceTensorRole::GatedNorm),
        "mixer.out_proj.weight" => state_space(StateSpaceTensorRole::OutputProjection),
        "mixer.gate.e_score_correction_bias" => LayerTensorRole::RouterCorrectionBias,
        "mixer.fc1_latent_proj.weight" => latent(LatentProjectionRole::Input),
        "mixer.fc2_latent_proj.weight" => latent(LatentProjectionRole::Output),
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

const fn state_space(tensor: StateSpaceTensorRole) -> LayerTensorRole {
    LayerTensorRole::StateSpace { tensor }
}

const fn latent(projection: LatentProjectionRole) -> LayerTensorRole {
    LayerTensorRole::LatentProjection { projection }
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
        .or_else(|| suffix.strip_prefix("mlp.experts."))
        .or_else(|| suffix.strip_prefix("mixer.experts."))?;
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
