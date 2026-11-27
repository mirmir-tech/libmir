//! Small BF16 checkpoints with per-expert tensors, written as safetensors.

use std::path::Path;

use serde_json::{Value, json};

pub const VOCAB: usize = 64;
const HIDDEN: usize = 32;

pub struct Checkpoint {
    config: Value,
    tensors: Vec<(String, Vec<usize>)>,
}

impl Checkpoint {
    fn add(&mut self, name: &str, shape: &[usize]) {
        self.tensors.push((name.to_owned(), shape.to_vec()));
    }

    fn experts(&mut self, mlp: &str, count: usize, width: usize) {
        for expert in 0..count {
            for (projection, shape) in
                [("gate", [width, HIDDEN]), ("up", [width, HIDDEN]), ("down", [HIDDEN, width])]
            {
                self.add(&format!("{mlp}.experts.{expert}.{projection}_proj.weight"), &shape);
            }
        }
    }

    fn boundary(&mut self) {
        self.add("model.embed_tokens.weight", &[VOCAB, HIDDEN]);
        self.add("model.norm.weight", &[HIDDEN]);
        self.add("lm_head.weight", &[VOCAB, HIDDEN]);
    }

    /// The config and the weights: norms at one, every other weight small
    /// and varied, so each expert computes something of its own.
    pub fn write(&self, root: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(root)?;
        std::fs::write(root.join("config.json"), serde_json::to_vec(&self.config)?)?;
        std::fs::write(root.join("tokenizer.json"), serde_json::to_vec(&tokenizer())?)?;
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        for (index, (name, shape)) in self.tensors.iter().enumerate() {
            let count = shape.iter().product::<usize>();
            let start = payload.len();
            for element in 0..count {
                let value = if name.ends_with("norm.weight") {
                    1.0
                } else {
                    let mixed = (element * 2_654_435_761 + index * 40_503) % 1_000;
                    (f32::from(u16::try_from(mixed).unwrap_or(0)) / 1_000.0 - 0.5) * 0.2
                };
                payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
            header.insert(
                name.clone(),
                json!({ "dtype": "BF16", "shape": shape, "data_offsets": [start, payload.len()] }),
            );
        }
        let mut header = serde_json::to_vec(&header)?;
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        std::fs::write(root.join("model.safetensors"), file)
    }
}

/// A word-level vocabulary of `VOCAB` words.
fn tokenizer() -> Value {
    let vocab = (0..VOCAB)
        .map(|id| (format!("w{id}"), json!(id)))
        .collect::<serde_json::Map<_, _>>();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null, "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "w0" }
    })
}

pub fn qwen3_moe() -> Checkpoint {
    let mut checkpoint = Checkpoint {
        config: json!({
            "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
            "hidden_size": HIDDEN, "intermediate_size": 16, "moe_intermediate_size": 8,
            "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 8, "vocab_size": VOCAB, "num_experts": 4, "num_experts_per_tok": 2,
            "norm_topk_prob": true, "decoder_sparse_step": 1, "mlp_only_layers": [],
            "hidden_act": "silu", "rms_norm_eps": 1e-6, "tie_word_embeddings": false,
            "max_position_embeddings": 256, "torch_dtype": "bfloat16"
        }),
        tensors: Vec::new(),
    };
    checkpoint.boundary();
    let layer = "model.layers.0";
    checkpoint.add(&format!("{layer}.input_layernorm.weight"), &[HIDDEN]);
    checkpoint.add(&format!("{layer}.post_attention_layernorm.weight"), &[HIDDEN]);
    for (projection, shape) in
        [("q", [32, HIDDEN]), ("k", [16, HIDDEN]), ("v", [16, HIDDEN]), ("o", [HIDDEN, 32])]
    {
        checkpoint.add(&format!("{layer}.self_attn.{projection}_proj.weight"), &shape);
    }
    checkpoint.add(&format!("{layer}.self_attn.q_norm.weight"), &[8]);
    checkpoint.add(&format!("{layer}.self_attn.k_norm.weight"), &[8]);
    checkpoint.add(&format!("{layer}.mlp.gate.weight"), &[4, HIDDEN]);
    checkpoint.experts(&format!("{layer}.mlp"), 4, 8);
    checkpoint
}

pub fn deepseek_v2() -> Checkpoint {
    let mut checkpoint = Checkpoint {
        config: json!({
            "architectures": ["DeepseekV2ForCausalLM"], "model_type": "deepseek_v2",
            "hidden_size": HIDDEN, "intermediate_size": 16, "moe_intermediate_size": 8,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
            "vocab_size": VOCAB, "q_lora_rank": null, "kv_lora_rank": 16,
            "qk_nope_head_dim": 8, "qk_rope_head_dim": 4, "v_head_dim": 8,
            "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 2,
            "first_k_dense_replace": 1, "moe_layer_freq": 1, "n_group": 1, "topk_group": 1,
            "scoring_func": "softmax", "topk_method": "greedy", "norm_topk_prob": false,
            "routed_scaling_factor": 1.0, "hidden_act": "silu", "rms_norm_eps": 1e-6,
            "max_position_embeddings": 256, "torch_dtype": "bfloat16"
        }),
        tensors: Vec::new(),
    };
    checkpoint.boundary();
    for layer in 0..2 {
        let prefix = format!("model.layers.{layer}");
        let attention = format!("{prefix}.self_attn");
        checkpoint.add(&format!("{prefix}.input_layernorm.weight"), &[HIDDEN]);
        checkpoint.add(&format!("{prefix}.post_attention_layernorm.weight"), &[HIDDEN]);
        checkpoint.add(&format!("{attention}.q_proj.weight"), &[4 * 12, HIDDEN]);
        checkpoint.add(&format!("{attention}.kv_a_proj_with_mqa.weight"), &[20, HIDDEN]);
        checkpoint.add(&format!("{attention}.kv_a_layernorm.weight"), &[16]);
        checkpoint.add(&format!("{attention}.kv_b_proj.weight"), &[4 * 16, 16]);
        checkpoint.add(&format!("{attention}.o_proj.weight"), &[HIDDEN, 4 * 8]);
        let mlp = format!("{prefix}.mlp");
        let dense = [("gate", [16, HIDDEN]), ("up", [16, HIDDEN]), ("down", [HIDDEN, 16])];
        if layer == 0 {
            for (projection, shape) in dense {
                checkpoint.add(&format!("{mlp}.{projection}_proj.weight"), &shape);
            }
            continue;
        }
        checkpoint.add(&format!("{mlp}.gate.weight"), &[4, HIDDEN]);
        for (projection, shape) in dense {
            checkpoint.add(&format!("{mlp}.shared_experts.{projection}_proj.weight"), &shape);
        }
        checkpoint.experts(&mlp, 4, 8);
    }
    checkpoint
}
