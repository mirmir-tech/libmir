use crate::{Error, Result};

pub(super) fn merge_format(
    current: &mut Option<(usize, usize)>,
    next: Option<(usize, usize)>,
    role: &'static str,
) -> Result<()> {
    if let Some(next) = next {
        if current.is_some_and(|current| current != next) {
            return Err(Error::UnsupportedDecoderLayer(format!(
                "CUDA {role} projections use different affine storage"
            )));
        }
        *current = Some(next);
    }
    Ok(())
}

pub(super) fn mixed_storage() -> Error {
    Error::UnsupportedDecoderLayer(
        "CUDA shared-routed router and experts use different checkpoint storage".into(),
    )
}
