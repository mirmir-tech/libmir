use super::{PagedStore, lock};
use crate::engine::{Array, Error, Result};

impl PagedStore {
    pub(in crate::engine::kv) fn validate_update(
        &self,
        keys: &Array,
        values: &Array,
    ) -> Result<()> {
        let shape = keys.native().shape()?;
        let dtype = keys.native().dtype()?;
        let dimensions = shape.dimensions();
        if dimensions.len() != 4
            || dimensions.contains(&0)
            || dimensions[0] != 1
            || shape != values.native().shape()?
            || dtype != values.native().dtype()?
            || !matches!(
                dtype,
                mirtal::DType::Float16 | mirtal::DType::Bfloat16 | mirtal::DType::Float32
            )
        {
            return Err(Error::InvalidModel(
                "paged K/V updates require matching nonempty floating-point arrays with batch one"
                    .into(),
            ));
        }
        if let Some(storage) = &self.storage {
            let arena = lock(&storage.arena)?;
            if dimensions[1] != arena.kv_heads
                || dimensions[3] != arena.head_dim
                || dtype != storage.input_dtype
            {
                return Err(Error::InvalidModel(
                    "paged K/V update changed the cached layout or input dtype".into(),
                ));
            }
        }
        Ok(())
    }
}
