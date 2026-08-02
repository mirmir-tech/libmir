use mircuda::{DeviceBuffer, bf16};

use crate::{AffineQuantizedWeight, CudaBackend, Error, Result, backend::linear::AffineProjection};

/// Single-token affine Int4/Int8 language-model output projection.
#[derive(Debug)]
pub struct CudaAffineOutputHead {
    projection: AffineProjection,
    weight: AffineQuantizedWeight,
    hidden_size: usize,
    vocab_size: usize,
}

impl CudaAffineOutputHead {
    pub fn from_weight(
        backend: &CudaBackend,
        hidden_size: usize,
        vocab_size: usize,
        weight: &AffineQuantizedWeight,
    ) -> Result<Self> {
        let config = weight.infer_config(1, hidden_size, vocab_size)?;
        Self::new(backend, hidden_size, vocab_size, config.group_size, config.bits, weight)
    }

    pub fn new(
        backend: &CudaBackend,
        hidden_size: usize,
        vocab_size: usize,
        group_size: usize,
        bits: usize,
        weight: &AffineQuantizedWeight,
    ) -> Result<Self> {
        weight.validate(1, hidden_size, vocab_size, group_size, bits)?;
        Ok(Self {
            projection: AffineProjection::new(
                backend, 1, hidden_size, vocab_size, group_size, bits, weight,
            )?,
            weight: weight.clone(),
            hidden_size,
            vocab_size,
        })
    }

    /// Enqueues full-vocabulary logits without host synchronization.
    pub fn execute(
        &self,
        input: &DeviceBuffer<bf16>,
        logits: &mut DeviceBuffer<bf16>,
    ) -> Result<()> {
        if input.len() != self.hidden_size || logits.len() != self.vocab_size {
            return Err(Error::InvalidDecoderKernel("affine output-head buffer mismatch"));
        }
        self.projection.execute(input, &self.weight, logits)
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use std::{fs, path::PathBuf};

    use mircuda::bf16;
    use models::weights::TensorInfo;

    use super::*;
    use crate::CudaConfig;

    const HIDDEN: usize = 64;
    const VOCAB: usize = 2;

    #[test]
    fn executes_every_native_mlx_affine_output_head() -> Result<()> {
        for bits in [2, 3, 4, 5, 6, 8] {
            check(bits)?;
        }
        Ok(())
    }

    fn check(bits: usize) -> Result<()> {
        let (path, infos) = fixture(bits)?;
        let backend = CudaBackend::new(CudaConfig::default())?;
        let mut upload = backend.begin_tensor_upload();
        for info in &infos {
            upload.enqueue(info)?;
        }
        let tensors = upload.finish()?;
        let weight = AffineQuantizedWeight::load(&tensors, "head")?;
        let operation = CudaAffineOutputHead::from_weight(&backend, HIDDEN, VOCAB, &weight)?;
        let input = copy(&backend, &[bf16::ONE; HIDDEN])?;
        let mut output =
            backend.inner.pool.allocate_zeroed::<bf16>(&backend.inner.stream, VOCAB)?;
        operation.execute(&input, &mut output)?;
        let mut host = backend.inner.context.allocate_pinned::<bf16>(VOCAB)?;
        backend.inner.stream.copy_to_host(&output, &mut host)?;
        assert_eq!(host.to_vec()?, [64.0, 128.0].map(bf16::from_f32));
        fs::remove_file(path)?;
        Ok(())
    }

    fn fixture(bits: usize) -> Result<(PathBuf, [TensorInfo; 3])> {
        let path = std::env::temp_dir()
            .join(format!("libmir-cuda-output-affine-{bits}-{}.bin", std::process::id()));
        let words = HIDDEN * bits / 32;
        let mut bytes = Vec::new();
        append_packed_row(&mut bytes, 1, bits);
        append_packed_row(&mut bytes, 2, bits);
        let weight_end = u64::try_from(bytes.len())?;
        append_bf16(&mut bytes, &[1.0; VOCAB]);
        let scale_end = u64::try_from(bytes.len())?;
        append_bf16(&mut bytes, &[0.0; VOCAB]);
        let bias_end = u64::try_from(bytes.len())?;
        fs::write(&path, bytes)?;
        Ok((
            path.clone(),
            [
                info("head.weight", &path, "U32", vec![VOCAB, words], 0, weight_end),
                info("head.scales", &path, "BF16", vec![VOCAB, 1], weight_end, scale_end),
                info("head.biases", &path, "BF16", vec![VOCAB, 1], scale_end, bias_end),
            ],
        ))
    }

    fn append_packed_row(bytes: &mut Vec<u8>, value: u32, bits: usize) {
        let mut words = vec![0_u32; HIDDEN * bits / 32];
        for index in 0..HIDDEN {
            let bit = index * bits;
            words[bit / 32] |= value << (bit % 32);
            if bit % 32 + bits > 32 {
                words[bit / 32 + 1] |= value >> (32 - bit % 32);
            }
        }
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }

    fn append_bf16(bytes: &mut Vec<u8>, values: &[f32]) {
        for value in values.iter().copied().map(bf16::from_f32) {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }

    fn info(
        name: &str,
        path: &std::path::Path,
        dtype: &str,
        shape: Vec<usize>,
        start: u64,
        end: u64,
    ) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            file: path.into(),
            dtype: dtype.into(),
            shape,
            data_start: 0,
            data_offsets: [start, end],
        }
    }

    fn copy(backend: &CudaBackend, values: &[bf16]) -> Result<DeviceBuffer<bf16>> {
        let mut host = backend.inner.context.allocate_pinned(values.len())?;
        host.copy_from_slice(values)?;
        let mut device = backend.inner.pool.allocate(&backend.inner.stream, values.len())?;
        backend.inner.stream.copy_to_device(&mut host, &mut device)?;
        Ok(device)
    }
}
