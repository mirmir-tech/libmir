//! Loaded models whose K/V caches share one accelerator budget.

use std::sync::{Arc, Mutex, Weak};

use super::{KvSizing, Model, ModelInner, automatic_cache::MeasuredKvBudget};
use crate::{Result, RuntimeError};

/// Tokens a model should be able to serve in one request before other
/// models are asked to give up pages.
pub(super) const MINIMUM_SHARE_TOKENS: u64 = 32_768;

#[derive(Clone, Debug, Default)]
pub(in crate::model) struct ModelRegistry {
    models: Arc<Mutex<Vec<Weak<ModelInner>>>>,
}

impl ModelRegistry {
    pub(in crate::model) fn register(&self, model: &Arc<ModelInner>) -> Result<()> {
        self.models.lock().map_or_else(
            |_| Err(poisoned()),
            |mut models| {
                models.retain(|entry| entry.strong_count() > 0);
                models.push(Arc::downgrade(model));
                Ok(())
            },
        )
    }

    /// Loaded models other than `except`, in load order.
    pub(super) fn others(&self, except: &Arc<ModelInner>) -> Result<Vec<Model>> {
        let mut models = self.models.lock().map_err(|_| poisoned())?;
        models.retain(|entry| entry.strong_count() > 0);
        Ok(models
            .iter()
            .filter_map(Weak::upgrade)
            .filter(|inner| !Arc::ptr_eq(inner, except))
            .map(|inner| Model { inner })
            .collect())
    }

    /// Every loaded model, in load order.
    pub(in crate::model) fn all(&self) -> Result<Vec<Model>> {
        let mut models = self.models.lock().map_err(|_| poisoned())?;
        models.retain(|entry| entry.strong_count() > 0);
        Ok(models.iter().filter_map(Weak::upgrade).map(|inner| Model { inner }).collect())
    }
}

fn poisoned() -> crate::Error {
    RuntimeError::Config("model registry lock is poisoned".into()).into()
}

impl Model {
    /// The measured budget; `None` for fixed or provisional caches.
    pub(super) fn measured_budget(&self) -> Result<Option<MeasuredKvBudget>> {
        Ok(match self.kv_sizing()? {
            KvSizing::Measured(budget) => Some(budget),
            KvSizing::Fixed | KvSizing::Provisional { .. } => None,
        })
    }
}
