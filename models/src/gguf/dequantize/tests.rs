use super::*;

#[test]
fn converts_halves_exactly() {
    for (bits, value) in [(0x3C00_u16, 1.0_f32), (0xC000, -2.0), (0x0001, 2f32.powi(-24)), (0, 0.0)]
    {
        assert_eq!(half(&bits.to_le_bytes()).to_bits(), value.to_bits());
    }
}

#[test]
fn decodes_q8_0_and_q4_0_blocks() -> Result<()> {
    let mut q8 = 0x3800_u16.to_le_bytes().to_vec();
    q8.extend((0..32).map(|value: u8| value.wrapping_sub(16)));
    let values = dequantize(GgmlType::Q8_0, &q8, 32)?;
    assert_eq!(values[0].to_bits(), (-8.0_f32).to_bits());
    assert_eq!(values[31].to_bits(), 7.5_f32.to_bits());

    let mut q4 = 0x4000_u16.to_le_bytes().to_vec();
    q4.extend([0x9F_u8; 16]);
    let values = dequantize(GgmlType::Q4_0, &q4, 32)?;
    assert_eq!(
        (values[0].to_bits(), values[16].to_bits()),
        (14.0_f32.to_bits(), 2.0_f32.to_bits())
    );
    Ok(())
}

#[test]
fn rejects_partial_rows_and_unknown_types() {
    assert!(dequantize(GgmlType::Q8_0, &[0; 33], 32).is_err());
    assert!(dequantize(GgmlType::F64, &[0; 8], 1).is_err());
}

/// Index-weighted sums of one patterned block, from gguf-py's decoders.
#[test]
fn low_bit_blocks_match_gguf_py() -> Result<()> {
    for (ggml_type, scale, expected) in [
        (GgmlType::Q1_0, Some(0), -16.625),
        (GgmlType::Q2_0, Some(0), 25.406_25),
        (GgmlType::TQ1_0, Some(52), -4.093_75),
        (GgmlType::TQ2_0, Some(64), 538.906_25),
        (GgmlType::NVFP4, None, 375.015_625),
    ] {
        let bytes = usize::try_from(ggml_type.block_bytes())?;
        let mut block = (0..bytes)
            .map(|i| u8::try_from((i * 37 + 11) % 256))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        match scale {
            // 0.03125 in half precision.
            Some(at) => block[at..at + 2].copy_from_slice(&0x2800_u16.to_le_bytes()),
            None => block[..4].copy_from_slice(&[0x30, 0x38, 0x29, 0x41]),
        }
        let columns = usize::try_from(ggml_type.block_elements())?;
        let values = dequantize(ggml_type, &block, columns)?;
        let weighted = values
            .iter()
            .enumerate()
            .map(|(i, value)| f64::from(*value) * f64::from(u32::try_from(i + 1).unwrap_or(0)))
            .sum::<f64>();
        assert_eq!((values.len(), weighted), (columns, expected), "{ggml_type}");
    }
    Ok(())
}
