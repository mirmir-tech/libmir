use std::sync::Mutex;

use super::History;
use crate::engine::{Array, Error, Result, Stream};
mod geometry;
use geometry::Geometry;
#[cfg(test)]
mod tests;

mirtal::metal_library! {
    fn append_library {
        name: "mirmir_history_append",
        source: file "kernels/history/append.metal",
    }
}

#[derive(Debug)]
pub struct Append {
    library: mirtal::MetalLibrary,
    plans: Mutex<[Option<mirtal::PreparedAliasing<5, 2>>; 3]>,
    /// Every layer of a decode step needs the same mask, so one is enough.
    padding_mask: Mutex<Option<(Vec<usize>, Array)>>,
}

impl Append {
    pub fn new() -> Result<Self> {
        Ok(Self {
            library: append_library()?,
            plans: Mutex::new([None, None, None]),
            padding_mask: Mutex::new(None),
        })
    }

    pub(super) fn execute(
        &self,
        history: &History,
        updates: [&Array; 2],
        stream: &Stream,
    ) -> Result<[Array; 2]> {
        let geometry = Geometry::new(history.shape, &history.tokens)?;
        let offsets = history
            .tokens
            .iter()
            .map(|tokens| Ok(u32::try_from(*tokens)?))
            .collect::<Result<Vec<_>>>()?;
        let offsets = mirtal::Array::from_slice(&offsets, [offsets.len()])?;
        let dtype = history.buffers[0].native().dtype()?;
        let index = match dtype {
            mirtal::DType::Float32 => 0,
            mirtal::DType::Float16 => 1,
            mirtal::DType::Bfloat16 => 2,
            _ => return Err(Error::InvalidModel("unsupported persistent history dtype".into())),
        };
        let mut plans = self.plans.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let plan = &mut plans[index];
        if plan.is_none() {
            *plan = Some(
                self.library
                    .export("history_append")?
                    .with_dtype_suffix(dtype)?
                    .prepare_aliasing(geometry.dispatch())?,
            );
        }
        let plan = plan.as_mut().ok_or(Error::NullHandle("history append plan"))?;
        plan.rebind(&geometry.constants, geometry.grid, geometry.group)?;
        let [keys, values] = plan.dispatch(
            stream.native(),
            [
                history.buffers[0].native(),
                history.buffers[1].native(),
                updates[0].native(),
                updates[1].native(),
                &offsets,
            ],
        )?;
        drop(plans);
        Ok([Array::from_native(keys)?, Array::from_native(values)?])
    }

    /// Returns an additive `[B, 1, 1, T]` mask with `-inf` over padded slots.
    pub(super) fn padding_mask(&self, tokens: &[usize]) -> Result<Array> {
        let cached = self
            .padding_mask
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .filter(|(cached, _)| cached == tokens)
            .map(|(_, mask)| mask.native().clone());
        if let Some(mask) = cached {
            return Array::from_native(mask);
        }
        let longest = tokens.iter().copied().max().unwrap_or(0);
        let mut values = Vec::with_capacity(tokens.len() * longest);
        for row in tokens {
            values.extend((0..longest).map(|slot| {
                if slot < *row {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            }));
        }
        let shape = [i32::try_from(tokens.len())?, 1, 1, i32::try_from(longest)?];
        let mask = Array::from_f32(&values, &shape)?;
        let retained = Array::from_native(mask.native().clone())?;
        *self.padding_mask.lock().unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some((tokens.to_vec(), retained));
        Ok(mask)
    }
}
