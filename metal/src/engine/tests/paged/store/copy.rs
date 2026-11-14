use crate::engine::{Array, Result, Stream};

#[test]
fn copies_only_the_target_page_for_every_native_dtype() -> Result<()> {
    let stream = Stream::new_gpu()?;
    for dtype in [mirtal::DType::Float32, mirtal::DType::Float16, mirtal::DType::Bfloat16] {
        let original = (0_u16..48).map(f32::from).collect::<Vec<_>>();
        let keys = Array::from_f32(&original, &[2, 4, 2, 3])?;
        let values = Array::from_f32(
            &original.iter().map(|value| -value).collect::<Vec<_>>(),
            &[2, 4, 2, 3],
        )?;
        let graph = stream.native().graph();
        let keys = graph.astype(keys.native(), dtype)?;
        let values = graph.astype(values.native(), dtype)?;
        let [keys, values] =
            stream.kernels().copy_kv_page(stream.native(), [&keys, &values], 0, 3)?;
        let [keys, values] =
            stream.kernels().copy_kv_page(stream.native(), [&keys, &values], 3, 1)?;
        let mut expected = original;
        for head in 0..2 {
            expected.copy_within(head * 24..head * 24 + 6, head * 24 + 18);
            expected.copy_within(head * 24..head * 24 + 6, head * 24 + 6);
        }
        assert_eq!(Array::from_native(keys)?.to_vec_f32(&stream)?, expected);
        assert_eq!(
            Array::from_native(values)?.to_vec_f32(&stream)?,
            expected.iter().map(|value| -value).collect::<Vec<_>>()
        );
    }
    Ok(())
}

#[test]
fn copies_quantized_words_without_changing_bits() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let original = (0..32).map(|value| 0xff00_aa00_u32 + value).collect::<Vec<_>>();
    let keys = Array::from_u32(&original, &[2, 4, 2, 2])?;
    let values = Array::from_u32(&original, &[2, 4, 2, 2])?;
    let [keys, values] =
        stream
            .kernels()
            .copy_kv_page(stream.native(), [keys.native(), values.native()], 3, 0)?;
    let mut expected = original;
    expected.copy_within(12..16, 0);
    expected.copy_within(28..32, 16);
    assert_eq!(Array::from_native(keys)?.to_vec_u32(&stream)?, expected);
    assert_eq!(Array::from_native(values)?.to_vec_u32(&stream)?, expected);
    Ok(())
}

#[test]
fn rejects_invalid_page_copy_indices() -> Result<()> {
    let stream = Stream::new_gpu()?;
    let keys = Array::from_f32(&[1.0; 8], &[1, 4, 2])?;
    let inputs = [keys.native(), keys.native()];
    assert!(stream.kernels().copy_kv_page(stream.native(), inputs, 1, 1).is_err());
    assert!(stream.kernels().copy_kv_page(stream.native(), inputs, 0, 4).is_err());
    Ok(())
}
