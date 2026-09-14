use super::KvCache;
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
        for (stored, update, shape) in [(&self.keys, keys, &ks), (&self.values, values, &vs)] {
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
        if let Some(pages) = self.pages.as_ref().filter(|pages| !pages.active()) {
            pages.validate_update(keys, values)?;
        }
        Ok(ks.dimensions()[2])
    }
}

#[cfg(test)]
mod tests;
