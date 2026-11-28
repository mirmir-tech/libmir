use super::{KvCache, KvStorage};
use crate::engine::{Array, Error, Result};

impl KvCache {
    pub(super) fn validate_update(&self, keys: &Array, values: &Array) -> Result<usize> {
        let ks = keys.native().shape()?;
        let vs = values.native().shape()?;
        if ks.dimensions().len() != 4
            || vs.dimensions().len() != 4
            || ks.dimensions().contains(&0)
            || vs.dimensions().contains(&0)
            || ks.dimensions()[..3] != vs.dimensions()[..3]
        {
            return Err(Error::InvalidModel("K/V updates require nonempty rank-four arrays with matching batch, heads and tokens".into()));
        }
        match &self.storage {
            KvStorage::Contiguous(store) => {
                validate_layout(
                    store.keys.as_ref(),
                    store.values.as_ref(),
                    keys,
                    values,
                    &ks,
                    &vs,
                )?;
            },
            KvStorage::Sliding(store) => {
                validate_layout(
                    store.keys.as_ref(),
                    store.values.as_ref(),
                    keys,
                    values,
                    &ks,
                    &vs,
                )?;
            },
            KvStorage::Paged(pages) => pages.validate_update(keys, values)?,
        }
        Ok(ks.dimensions()[2])
    }
}

fn validate_layout(
    stored_keys: Option<&Array>,
    stored_values: Option<&Array>,
    keys: &Array,
    values: &Array,
    ks: &mirtal::Shape,
    vs: &mirtal::Shape,
) -> Result<()> {
    for (stored, update, shape) in [(stored_keys, keys, ks), (stored_values, values, vs)] {
        if let Some(stored) = stored {
            let current = stored.native().shape()?;
            if [0, 1, 3]
                .into_iter()
                .any(|axis| current.dimensions()[axis] != shape.dimensions()[axis])
                || stored.native().dtype()? != update.native().dtype()?
            {
                return Err(Error::InvalidModel(
                    "K/V update changed the cached array layout or dtype".into(),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
