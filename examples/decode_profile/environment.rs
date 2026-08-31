use std::env;

pub(super) fn argument(index: usize, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    env::args().nth(index).map_or(Ok(default), |value| Ok(value.parse()?))
}

pub(super) fn environment_usize(
    name: &str,
    default: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    env::var(name).map_or(Ok(default), |value| Ok(value.parse()?))
}

pub(super) fn environment_u64(name: &str, default: u64) -> Result<u64, Box<dyn std::error::Error>> {
    env::var(name).map_or(Ok(default), |value| Ok(value.parse()?))
}

#[cfg(feature = "cuda")]
pub(super) fn environment_optional_usize(
    name: &str,
) -> Result<Option<usize>, Box<dyn std::error::Error>> {
    env::var(name).map_or(Ok(None), |value| Ok(Some(value.parse()?)))
}

pub(super) fn enabled(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1" | "true" | "TRUE" | "yes" | "YES"))
}

#[cfg(feature = "metal")]
pub(super) fn disabled(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("0" | "false" | "FALSE" | "no" | "NO"))
}
