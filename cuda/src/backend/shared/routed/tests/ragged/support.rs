use super::*;

pub(super) fn compare_continuation(
    backend: &CudaBackend,
    actual: &mut [CudaSharedRoutedModelSession],
    reference: &mut [CudaSharedRoutedModelSession],
    tables: &mut [BlockTable],
) -> Result<()> {
    for row in 0..actual.len() {
        tables[row].set_token_len(actual[row].position() + 1);
        let token = u32::try_from(row + 1)?;
        let expected = read(backend, reference[row].decode(Uuid::nil(), token, &tables[row])?)?;
        let values = read(backend, actual[row].decode(Uuid::nil(), token, &tables[row])?)?;
        assert!(expected.iter().any(|value| value.to_f32().abs() > 0.01));
        for (actual, expected) in values.iter().zip(&expected) {
            assert!(
                (actual.to_f32() - expected.to_f32()).abs() < 0.03,
                "row={row} actual={actual:?} expected={expected:?}"
            );
        }
    }
    Ok(())
}
