use super::{Arena, Array, Result, Stream};

pub(super) fn copy_page(
    arena: &mut Arena,
    source: usize,
    target: usize,
    stream: &Stream,
) -> Result<()> {
    let [keys, values] = stream.kernels().copy_kv_page(
        stream.native(),
        [arena.keys.native(), arena.values.native()],
        source,
        target,
    )?;
    arena.keys = Array::from_native(keys)?;
    arena.values = Array::from_native(values)?;
    if let (Some(keys), Some(values)) = (&arena.key_scales, &arena.value_scales) {
        let [keys, values] = stream.kernels().copy_kv_page(
            stream.native(),
            [keys.native(), values.native()],
            source,
            target,
        )?;
        arena.key_scales = Some(Array::from_native(keys)?);
        arena.value_scales = Some(Array::from_native(values)?);
    }
    Ok(())
}
