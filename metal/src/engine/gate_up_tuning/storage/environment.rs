use std::{fs::File, io, sync::OnceLock};

use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Eq, Deserialize, Serialize)]
pub(super) struct Environment {
    mlx_version: String,
    device_name: String,
    executable_hash: [u8; 32],
}

impl Environment {
    pub(super) fn current() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            mlx_version: mirtal::version()?,
            device_name: mirtal::Device::gpu(0).name()?,
            executable_hash: executable_hash()?,
        })
    }
}

// A crate version does not identify local kernel/plan changes. Hash the linked
// implementation once per process, outside inference, without requiring Git or
// sources on the deployment machine. An unreadable executable disables reuse.
fn executable_hash() -> io::Result<[u8; 32]> {
    static HASH: OnceLock<Option<[u8; 32]>> = OnceLock::new();
    HASH.get_or_init(|| hash_executable().ok())
        .ok_or_else(|| io::Error::other("cannot identify Metal executable for tuning cache"))
}

fn hash_executable() -> io::Result<[u8; 32]> {
    let mut executable = File::open(std::env::current_exe()?)?;
    let mut hash = blake3::Hasher::new();
    hash.update_reader(&mut executable)?;
    Ok(*hash.finalize().as_bytes())
}
