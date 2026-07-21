use std::{fs, path::PathBuf};

use mircuda::bf16;
use models::{
    layout::DecoderConfig,
    weights::{TensorCatalog, TensorInfo},
};

use crate::{Error, Result};

const ROOT: &str = "language_model.model";
const GROUP: usize = 64;
const BITS: usize = 4;

pub(super) struct HybridFixture {
    path: PathBuf,
    bytes: Vec<u8>,
    infos: Vec<TensorInfo>,
}

impl HybridFixture {
    pub(super) fn new(decoder: &DecoderConfig) -> Result<Self> {
        let path = std::env::temp_dir()
            .join(format!("libmir-cuda-shared-routed-{}.bin", std::process::id()));
        let mut fixture = Self {
            path,
            bytes: Vec::new(),
            infos: Vec::new(),
        };
        fixture.projection(
            &format!("{ROOT}.embed_tokens"),
            1,
            decoder.hidden_size,
            decoder.vocab_size,
            BITS,
        )?;
        fixture.projection(
            "language_model.lm_head",
            1,
            decoder.hidden_size,
            decoder.vocab_size,
            BITS,
        )?;
        fixture.norm(&format!("{ROOT}.norm.weight"), decoder.hidden_size)?;
        for layer in 0..decoder.num_hidden_layers {
            fixture.layer(decoder, layer)?;
        }
        fs::write(&fixture.path, &fixture.bytes)?;
        Ok(fixture)
    }

    pub(super) fn catalog(&self) -> TensorCatalog {
        TensorCatalog { tensors: self.infos.clone() }
    }

    fn layer(&mut self, decoder: &DecoderConfig, layer: usize) -> Result<()> {
        let prefix = format!("{ROOT}.layers.{layer}");
        self.norm(&format!("{prefix}.input_layernorm.weight"), decoder.hidden_size)?;
        self.norm(&format!("{prefix}.post_attention_layernorm.weight"), decoder.hidden_size)?;
        self.moe(decoder, &format!("{prefix}.mlp"))?;
        if layer == 0 {
            self.linear_attention(decoder, &format!("{prefix}.linear_attn"))
        } else {
            self.full_attention(decoder, &format!("{prefix}.self_attn"))
        }
    }

    fn moe(&mut self, decoder: &DecoderConfig, prefix: &str) -> Result<()> {
        let experts = decoder.num_experts.unwrap_or_default();
        let hidden = decoder.hidden_size;
        let routed = decoder.moe_intermediate_size.unwrap_or_default();
        let shared = decoder.shared_expert_intermediate_size.unwrap_or_default();
        self.projection(&format!("{prefix}.gate"), 1, hidden, experts, 8)?;
        for name in ["switch_mlp.gate_proj", "switch_mlp.up_proj"] {
            self.projection(&format!("{prefix}.{name}"), experts, hidden, routed, BITS)?;
        }
        self.projection(&format!("{prefix}.switch_mlp.down_proj"), experts, routed, hidden, BITS)?;
        for name in ["shared_expert.gate_proj", "shared_expert.up_proj"] {
            self.projection(&format!("{prefix}.{name}"), 1, hidden, shared, BITS)?;
        }
        self.projection(&format!("{prefix}.shared_expert.down_proj"), 1, shared, hidden, BITS)?;
        self.projection(&format!("{prefix}.shared_expert_gate"), 1, hidden, 1, 8)
    }

    fn linear_attention(&mut self, decoder: &DecoderConfig, prefix: &str) -> Result<()> {
        let linear = decoder.linear_attention.as_ref().ok_or_else(|| {
            Error::UnsupportedDecoderLayer("fixture is missing linear attention".into())
        })?;
        let key = linear.key_heads * linear.key_head_dim;
        let value = linear.value_heads * linear.value_head_dim;
        let mixed = 2 * key + value;
        for (name, output) in [
            ("in_proj_qkv", mixed),
            ("in_proj_z", value),
            ("in_proj_a", linear.value_heads),
            ("in_proj_b", linear.value_heads),
        ] {
            self.projection(&format!("{prefix}.{name}"), 1, decoder.hidden_size, output, BITS)?;
        }
        self.projection(&format!("{prefix}.out_proj"), 1, value, decoder.hidden_size, BITS)?;
        self.bf16(
            &format!("{prefix}.conv1d.weight"),
            vec![mixed, linear.convolution_kernel_size, 1],
            mixed * linear.convolution_kernel_size,
            0.0,
        )?;
        self.norm(&format!("{prefix}.norm.weight"), linear.value_head_dim)?;
        self.bf16(&format!("{prefix}.A_log"), vec![linear.value_heads], linear.value_heads, 0.0)?;
        self.bf16(&format!("{prefix}.dt_bias"), vec![linear.value_heads], linear.value_heads, 0.0)
    }

    fn full_attention(&mut self, decoder: &DecoderConfig, prefix: &str) -> Result<()> {
        let head = decoder.head_dim;
        let query = decoder.num_attention_heads * head;
        let key_value = decoder.num_key_value_heads * head;
        for (name, input, output) in [
            ("q_proj", decoder.hidden_size, 2 * query),
            ("k_proj", decoder.hidden_size, key_value),
            ("v_proj", decoder.hidden_size, key_value),
            ("o_proj", query, decoder.hidden_size),
        ] {
            self.projection(&format!("{prefix}.{name}"), 1, input, output, BITS)?;
        }
        self.norm(&format!("{prefix}.q_norm.weight"), head)?;
        self.norm(&format!("{prefix}.k_norm.weight"), head)
    }

    fn projection(
        &mut self,
        prefix: &str,
        matrices: usize,
        input: usize,
        output: usize,
        bits: usize,
    ) -> Result<()> {
        let packed = input / (32 / bits);
        let groups = input / GROUP;
        let shape = |tail| {
            if matrices == 1 {
                vec![output, tail]
            } else {
                vec![matrices, output, tail]
            }
        };
        self.u32(&format!("{prefix}.weight"), shape(packed), matrices * output * packed)?;
        self.bf16(&format!("{prefix}.scales"), shape(groups), matrices * output * groups, 0.0)?;
        self.bf16(&format!("{prefix}.biases"), shape(groups), matrices * output * groups, 0.0)
    }

    fn norm(&mut self, name: &str, elements: usize) -> Result<()> {
        self.bf16(name, vec![elements], elements, 1.0)
    }

    fn u32(&mut self, name: &str, shape: Vec<usize>, elements: usize) -> Result<()> {
        let start = u64::try_from(self.bytes.len())?;
        self.bytes.resize(self.bytes.len() + elements * 4, 0);
        self.info(name, "U32", shape, start)
    }

    fn bf16(&mut self, name: &str, shape: Vec<usize>, elements: usize, value: f32) -> Result<()> {
        let start = u64::try_from(self.bytes.len())?;
        for _ in 0..elements {
            self.bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        self.info(name, "BF16", shape, start)
    }

    fn info(&mut self, name: &str, dtype: &str, shape: Vec<usize>, start: u64) -> Result<()> {
        self.infos.push(TensorInfo {
            name: name.into(),
            file: self.path.clone(),
            dtype: dtype.into(),
            shape,
            data_start: 0,
            data_offsets: [start, u64::try_from(self.bytes.len())?],
        });
        Ok(())
    }
}

impl Drop for HybridFixture {
    fn drop(&mut self) {
        let _removed = fs::remove_file(&self.path);
    }
}
