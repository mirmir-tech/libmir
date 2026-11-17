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
    assert!(dequantize(GgmlType::IQ2_XXS, &[0; 66], 256).is_err());
}
