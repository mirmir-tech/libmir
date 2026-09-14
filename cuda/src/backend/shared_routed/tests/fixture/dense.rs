use super::*;

impl HybridFixture {
    pub(in crate::backend::shared_routed::tests) fn nonzero_dense(
        decoder: &DecoderConfig,
    ) -> Result<Self> {
        let mut fixture = Self::new(decoder)?;
        let mut bytes = Vec::new();
        let mut infos = Vec::new();
        for mut info in fixture.infos.clone() {
            if info.name.ends_with(".scales") || info.name.ends_with(".biases") {
                continue;
            }
            let data = if info.dtype == "U32" {
                let last = info.shape.len() - 1;
                info.shape[last] *= 8;
                info.dtype = "BF16".into();
                let seed = info.name.bytes().map(usize::from).sum::<usize>();
                (0..info.shape.iter().product::<usize>())
                    .flat_map(|i| {
                        let signed = i32::try_from((i * 17 + seed) % 101).unwrap_or_default() - 50;
                        #[allow(clippy::cast_precision_loss)]
                        let value = signed as f32 / 256.0;
                        bf16::from_f32(value).to_bits().to_le_bytes()
                    })
                    .collect::<Vec<_>>()
            } else if info.name.ends_with(".conv1d.weight") {
                (0..info.shape.iter().product::<usize>())
                    .flat_map(|i| {
                        bf16::from_f32(if i.is_multiple_of(2) {
                            0.125
                        } else {
                            -0.0625
                        })
                        .to_bits()
                        .to_le_bytes()
                    })
                    .collect()
            } else {
                fixture.bytes
                    [usize::try_from(info.data_offsets[0])?..usize::try_from(info.data_offsets[1])?]
                    .to_vec()
            };
            info.data_offsets[0] = u64::try_from(bytes.len())?;
            bytes.extend(data);
            info.data_offsets[1] = u64::try_from(bytes.len())?;
            infos.push(info);
        }
        fixture.bytes = bytes;
        fixture.infos = infos;
        fs::write(&fixture.path, &fixture.bytes)?;
        Ok(fixture)
    }
}
