use std::{fs, path::PathBuf};

use mircuda::bf16;
use models::weights::TensorInfo;

use super::super::AffineGatedFullAttentionConfig;
use crate::{CudaBackend, CudaTensorSet, Result};

pub(super) const PREFIX: &str = "language_model.model.layers.0.self_attn";

pub(super) struct AttentionFixture {
    path: PathBuf,
    bytes: Vec<u8>,
    infos: Vec<TensorInfo>,
}

impl AttentionFixture {
    pub(super) fn new(config: AffineGatedFullAttentionConfig) -> Result<Self> {
        let path = std::env::temp_dir()
            .join(format!("libmir-cuda-gated-attention-{}.bin", std::process::id()));
        let mut fixture = Self {
            path,
            bytes: Vec::new(),
            infos: Vec::new(),
        };
        let query = config.query_width()?;
        let key_value = config.key_value_width()?;
        fixture.projection("q_proj", config.hidden_size, 2 * query, config)?;
        fixture.projection("k_proj", config.hidden_size, key_value, config)?;
        fixture.projection("v_proj", config.hidden_size, key_value, config)?;
        fixture.projection("o_proj", query, config.hidden_size, config)?;
        fixture.bf16("q_norm.weight", vec![config.head_dim], vec![1.0; config.head_dim])?;
        fixture.bf16("k_norm.weight", vec![config.head_dim], vec![1.0; config.head_dim])?;
        fs::write(&fixture.path, &fixture.bytes)?;
        Ok(fixture)
    }

    pub(super) fn upload(&self, backend: &CudaBackend) -> Result<CudaTensorSet> {
        let mut upload = backend.begin_tensor_upload();
        for info in &self.infos {
            upload.enqueue(info)?;
        }
        upload.finish()
    }

    fn projection(
        &mut self,
        name: &str,
        input: usize,
        output: usize,
        config: AffineGatedFullAttentionConfig,
    ) -> Result<()> {
        let packed = input / (32 / config.bits);
        let groups = input / config.group_size;
        self.u32(&format!("{name}.weight"), vec![output, packed], output * packed)?;
        self.bf16(&format!("{name}.scales"), vec![output, groups], vec![0.0; output * groups])?;
        self.bf16(&format!("{name}.biases"), vec![output, groups], vec![0.0; output * groups])
    }

    fn u32(&mut self, name: &str, shape: Vec<usize>, elements: usize) -> Result<()> {
        let start = u64::try_from(self.bytes.len())?;
        self.bytes.resize(self.bytes.len() + elements * 4, 0);
        self.info(name, "U32", shape, start)
    }

    fn bf16(&mut self, name: &str, shape: Vec<usize>, values: Vec<f32>) -> Result<()> {
        let start = u64::try_from(self.bytes.len())?;
        for value in values {
            self.bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        self.info(name, "BF16", shape, start)
    }

    fn info(&mut self, name: &str, dtype: &str, shape: Vec<usize>, start: u64) -> Result<()> {
        self.infos.push(TensorInfo {
            name: format!("{PREFIX}.{name}"),
            file: self.path.clone(),
            dtype: dtype.into(),
            shape,
            data_start: 0,
            data_offsets: [start, u64::try_from(self.bytes.len())?],
        });
        Ok(())
    }
}

impl Drop for AttentionFixture {
    fn drop(&mut self) {
        let _removed = fs::remove_file(&self.path);
    }
}
