use super::{Array, Result};

impl Array {
    /// Creates an independent Rust owner for the same device-resident MLX
    /// array.
    pub fn snapshot(&self) -> Result<Self> {
        Self::from_native(self.native().clone())
    }
}
