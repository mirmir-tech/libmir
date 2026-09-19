use super::*;

const SCALE: f32 = 1.0 / 128.0;

impl HybridFixture {
    /// Keeps the affine-quantized routed layout, replacing its zero tensors
    /// with deterministic values centered around zero.
    pub(in crate::backend::shared_routed::tests) fn nonzero_routed(
        decoder: &DecoderConfig,
    ) -> Result<Self> {
        let mut fixture = Self::new(decoder)?;
        for info in fixture.infos.clone() {
            let start = usize::try_from(info.data_offsets[0])?;
            let end = usize::try_from(info.data_offsets[1])?;
            let seed = info.name.bytes().map(usize::from).sum::<usize>();
            let constant = if info.name.ends_with(".scales") {
                Some(SCALE)
            } else if info.name.ends_with(".biases") {
                Some(-7.5 * SCALE)
            } else {
                None
            };
            let bytes = &mut fixture.bytes[start..end];
            if info.dtype == "U32" {
                for (index, byte) in bytes.iter_mut().enumerate() {
                    *byte = u8::try_from((index * 37 + seed) % 251)?;
                }
            } else if let Some(value) = constant {
                for pair in bytes.as_chunks_mut::<2>().0 {
                    pair.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
                }
            } else if info.name.ends_with(".conv1d.weight") {
                for (index, pair) in bytes.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                    let value = if index.is_multiple_of(2) {
                        0.125
                    } else {
                        -0.0625
                    };
                    pair.copy_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
                }
            }
        }
        fs::write(&fixture.path, &fixture.bytes)?;
        Ok(fixture)
    }
}
