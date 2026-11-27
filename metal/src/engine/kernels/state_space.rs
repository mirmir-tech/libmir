use super::{Kernels, template};
use crate::engine::{Error, Result};

mirtal::metal_kernel! {
    fn state_space_decode {
        name: "mirmir_state_space_decode",
        templates: [
            T: dtype = bf16, U: dtype = f32, Dh: int = 64, Ds: int = 128, H: int = 64, G: int = 8,
        ],
        inputs: [X: T, A_log: float, B: T, C: T, D: float, dt: f32, state_in: U],
        outputs: [out: T, state_out: U],
        source: file "kernels/state_space_decode.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

pub(super) fn new_state_space_decode_kernel() -> Result<mirtal::MetalKernel<7, 2>> {
    Ok(state_space_decode()?)
}

impl Kernels {
    pub(crate) fn state_space_decode(
        &self,
        stream: &mirtal::Stream,
        inputs: [&mirtal::Array; 7],
    ) -> Result<[mirtal::Array; 2]> {
        let [x, a_log, b, c, d, dt, state] = inputs;
        let x_shape = x.shape()?;
        let b_shape = b.shape()?;
        let dims = x_shape.dimensions().to_vec();
        let groups = b_shape.dimensions().to_vec();
        if dims.len() != 4 || groups.len() != 4 || groups[0] != dims[0] || dims[1] != 1 {
            return Err(Error::InvalidModel("Mamba-2 decode shapes are incompatible".into()));
        }
        let [rows, _, heads, channels] = [dims[0], dims[1], dims[2], dims[3]];
        let state_size = groups[3];
        if b.shape()? != c.shape()?
            || a_log.len() != heads
            || d.len() != heads
            || dt.len() != rows * heads
            || state.shape()?.dimensions() != [rows, heads, channels, state_size]
            || [rows, heads, channels, groups[2], state_size].contains(&0)
            || !heads.is_multiple_of(groups[2])
            || !state_size.is_multiple_of(32)
        {
            return Err(Error::InvalidModel("Mamba-2 decode shapes are incompatible".into()));
        }
        Ok(self.state_space_decode.dispatch(
            stream,
            inputs,
            &[
                mirtal::OutputSpec::new(x_shape, x.dtype()?),
                mirtal::OutputSpec::new(state.shape()?, state.dtype()?),
            ],
            &mirtal::Dispatch::new([32, channels, heads * rows], [32, 8, 1]).templates([
                mirtal::TemplateArg::dtype("T", x.dtype()?),
                mirtal::TemplateArg::dtype("U", state.dtype()?),
                template("Dh", channels)?,
                template("Ds", state_size)?,
                template("H", heads)?,
                template("G", heads / groups[2])?,
            ]),
        )?)
    }
}
