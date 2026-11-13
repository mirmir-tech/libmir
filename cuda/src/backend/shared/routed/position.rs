use crate::{Error, Result};

pub(super) fn text_positions(start: usize, tokens: usize, delta: i32) -> Result<Vec<u32>> {
    let end = start
        .checked_add(tokens)
        .ok_or(Error::InvalidDecoderKernel("text position range overflow"))?;
    let values = (start..end)
        .map(|position| {
            let shifted = i64::try_from(position)? + i64::from(delta);
            Ok(u32::try_from(shifted)?)
        })
        .collect::<std::result::Result<Vec<_>, Error>>()?;
    Ok(values.repeat(3))
}
