use std::{fs, path::PathBuf};

use mircuda::bf16;
use models::weights::TensorInfo;

use super::super::AffineGatedDeltaLayerConfig;
use crate::{CudaBackend, CudaTensorSet, Result};

pub(super) const PREFIX: &str = "language_model.model.layers.0.linear_attn";

pub(super) struct LayerFixture {
    path: PathBuf,
    bytes: Vec<u8>,
    infos: Vec<TensorInfo>,
}

impl LayerFixture {
    pub(super) fn new(config: AffineGatedDeltaLayerConfig) -> Result<Self> {
        let path = std::env::temp_dir()
            .join(format!("libmir-cuda-gated-delta-{}.bin", std::process::id()));
        let mut fixture = Self {
            path,
            bytes: Vec::new(),
            infos: Vec::new(),
        };
        let mixed = 2 * config.key_heads * config.key_dim + config.value_heads * config.value_dim;
        let value = config.value_heads * config.value_dim;
        fixture.projection("in_proj_qkv", config.hidden_size, mixed, config)?;
        fixture.projection("in_proj_z", config.hidden_size, value, config)?;
        fixture.projection("in_proj_a", config.hidden_size, config.value_heads, config)?;
        fixture.projection("in_proj_b", config.hidden_size, config.value_heads, config)?;
        fixture.projection("out_proj", value, config.hidden_size, config)?;
        fixture.bf16(
            "conv1d.weight",
            vec![mixed, config.convolution_kernel_size, 1],
            vec![0.0; mixed * config.convolution_kernel_size],
        )?;
        fixture.bf16("norm.weight", vec![config.value_dim], vec![1.0; config.value_dim])?;
        fixture.bf16("A_log", vec![config.value_heads], vec![0.0; config.value_heads])?;
        fixture.bf16("dt_bias", vec![config.value_heads], vec![0.0; config.value_heads])?;
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
        config: AffineGatedDeltaLayerConfig,
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
        let end = u64::try_from(self.bytes.len())?;
        self.infos.push(TensorInfo {
            name: format!("{PREFIX}.{name}"),
            file: self.path.clone(),
            dtype: dtype.into(),
            shape,
            data_start: 0,
            data_offsets: [start, end],
        });
        Ok(())
    }
}

impl Drop for LayerFixture {
    fn drop(&mut self) {
        let _removed = fs::remove_file(&self.path);
    }
}
