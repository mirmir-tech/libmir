use std::{fs, path::PathBuf};

use mircuda::bf16;
use models::weights::TensorInfo;

use super::super::AffineSharedExpertMoeConfig;
use crate::{CudaBackend, CudaTensorSet, Result};

pub(super) const PREFIX: &str = "language_model.model.layers.0.mlp";

pub(super) struct MoeFixture {
    path: PathBuf,
    bytes: Vec<u8>,
    infos: Vec<TensorInfo>,
}

impl MoeFixture {
    pub(super) fn new(config: AffineSharedExpertMoeConfig) -> Result<Self> {
        let path =
            std::env::temp_dir().join(format!("libmir-cuda-shared-moe-{}.bin", std::process::id()));
        let mut fixture = Self {
            path,
            bytes: Vec::new(),
            infos: Vec::new(),
        };
        fixture.projection(
            "gate",
            1,
            config.hidden_size,
            config.expert_count,
            config.router_bits,
            config,
        )?;
        for name in ["switch_mlp.gate_proj", "switch_mlp.up_proj"] {
            fixture.projection(
                name,
                config.expert_count,
                config.hidden_size,
                config.routed_intermediate_size,
                config.expert_bits,
                config,
            )?;
        }
        fixture.projection(
            "switch_mlp.down_proj",
            config.expert_count,
            config.routed_intermediate_size,
            config.hidden_size,
            config.expert_bits,
            config,
        )?;
        for name in ["shared_expert.gate_proj", "shared_expert.up_proj"] {
            fixture.projection(
                name,
                1,
                config.hidden_size,
                config.shared_intermediate_size,
                config.expert_bits,
                config,
            )?;
        }
        fixture.projection(
            "shared_expert.down_proj",
            1,
            config.shared_intermediate_size,
            config.hidden_size,
            config.expert_bits,
            config,
        )?;
        fixture.projection(
            "shared_expert_gate",
            1,
            config.hidden_size,
            1,
            config.router_bits,
            config,
        )?;
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

    #[allow(clippy::too_many_arguments)]
    fn projection(
        &mut self,
        name: &str,
        matrices: usize,
        input: usize,
        output: usize,
        bits: usize,
        config: AffineSharedExpertMoeConfig,
    ) -> Result<()> {
        let packed = input / (32 / bits);
        let groups = input / config.group_size;
        let weight_shape = shape(matrices, output, packed);
        let group_shape = shape(matrices, output, groups);
        let rows = matrices * output;
        self.u32(&format!("{name}.weight"), weight_shape, rows * packed)?;
        self.bf16(&format!("{name}.scales"), group_shape.clone(), rows * groups)?;
        self.bf16(&format!("{name}.biases"), group_shape, rows * groups)
    }

    fn u32(&mut self, name: &str, shape: Vec<usize>, elements: usize) -> Result<()> {
        let start = u64::try_from(self.bytes.len())?;
        self.bytes.resize(self.bytes.len() + elements * 4, 0);
        self.info(name, "U32", shape, start)
    }

    fn bf16(&mut self, name: &str, shape: Vec<usize>, elements: usize) -> Result<()> {
        let start = u64::try_from(self.bytes.len())?;
        for _ in 0..elements {
            self.bytes.extend_from_slice(&bf16::ZERO.to_bits().to_le_bytes());
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

impl Drop for MoeFixture {
    fn drop(&mut self) {
        let _removed = fs::remove_file(&self.path);
    }
}

fn shape(matrices: usize, output: usize, trailing: usize) -> Vec<usize> {
    if matrices == 1 {
        vec![output, trailing]
    } else {
        vec![matrices, output, trailing]
    }
}
