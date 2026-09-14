#[cfg(test)]
mod probe;
#[cfg(test)]
mod tests;

use super::BoundLinear;
use crate::engine::{Array, Result, RouterOutput, Stream};

impl BoundLinear {
    pub(in crate::engine) fn route_unit(
        &self,
        input: &Array,
        top_k: i32,
        stream: &Stream,
    ) -> Result<RouterOutput> {
        #[cfg(test)]
        if stream.config().diagnostics.router_precision == crate::config::RouterPrecision::Float32 {
            return self.route_precise(input, top_k, stream);
        }
        self.forward(input, stream)?.router_top_k_unit(top_k, stream)
    }

    #[cfg(test)]
    fn route_precise(&self, input: &Array, top_k: i32, stream: &Stream) -> Result<RouterOutput> {
        let precise = Array::from_native(
            stream.native().graph().astype(input.native(), mirtal::DType::Float32)?,
        )?;
        let routing = self.forward(&precise, stream)?.router_top_k_unit(top_k, stream)?;
        // Only router precision changes; expert inputs and reduction weights keep model
        // dtype.
        Ok(RouterOutput {
            indices: routing.indices,
            weights: routing.weights.astype_like(input, stream)?,
        })
    }
}
