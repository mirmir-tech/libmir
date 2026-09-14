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
    plans: Mutex<[Option<mirtal::PreparedAliasing<4, 2>>; 3]>,
}

impl Append {
    pub fn new() -> Result<Self> {
        Ok(Self {
            library: append_library()?,
            plans: Mutex::new([None, None, None]),
        })
    }

    pub(super) fn execute(
        &self,
        history: &History,
        updates: [&Array; 2],
        stream: &Stream,
    ) -> Result<[Array; 2]> {
        let geometry = Geometry::new(history.shape, history.tokens)?;
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
            ],
        )?;
        drop(plans);
        Ok([Array::from_native(keys)?, Array::from_native(values)?])
    }
}
