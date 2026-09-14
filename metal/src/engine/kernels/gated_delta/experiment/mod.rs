use mirtal::{Array, DType, Dispatch, MetalKernel, OutputSpec, Stream, TemplateArg};

use super::super::{gated_delta_recurrence, template};
use crate::engine::Result;

pub(in crate::engine) mod decode;
mod tests;

thread_local! {
    static CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
pub(in crate::engine) fn calls() -> usize {
    CALLS.get()
}
pub(in crate::engine) fn record() {
    CALLS.set(CALLS.get() + 1);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::engine) enum Layout {
    Native,
    Explicit,
    Packed,
}

mirtal::metal_kernel! {
    fn explicit {
        name: "mirmir_gdn_explicit",
        templates: [
            InT: dtype = bf16, StT: dtype = f32, DK: int = 128, DV: int = 128,
            HK: int = 8, HV: int = 16, STEPS: int = 4,
        ],
        inputs: [query: InT, key: InT, value: InT, decay: f32, update: f32, state: StT],
        outputs: [output: InT, next_state: StT],
        source: file "kernels/gated_delta/explicit.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}
mirtal::metal_kernel! {
    fn packed {
        name: "mirmir_gdn_packed",
        templates: [
            InT: dtype = bf16, StT: dtype = f32, DK: int = 128, DV: int = 128,
            HK: int = 8, HV: int = 16, STEPS: int = 4,
        ],
        inputs: [query: InT, key: InT, value: InT, decay: f32, update: f32, state: StT],
        outputs: [output: InT, next_state: StT],
        source: file "kernels/gated_delta/packed.metal",
        header: inline "",
        row_contiguous: true,
        atomic_outputs: false,
    }
}

#[derive(Debug)]
pub(in crate::engine) struct Recurrence {
    native: MetalKernel<6, 2>,
    explicit: MetalKernel<6, 2>,
    packed: MetalKernel<6, 2>,
}

impl Recurrence {
    pub(in crate::engine) fn new() -> Result<Self> {
        Ok(Self {
            native: gated_delta_recurrence()?,
            explicit: explicit()?,
            packed: packed()?,
        })
    }

    pub(in crate::engine) fn run(
        &self,
        stream: &Stream,
        inputs: [&Array; 6],
        layout: Layout,
    ) -> Result<[Array; 2]> {
        let q = inputs[0].shape()?.dimensions().to_vec();
        let v = inputs[2].shape()?.dimensions().to_vec();
        super::validate_recurrence(inputs, &q, &v)?;
        let eligible =
            q[3] == 128 && v[3].is_multiple_of(8) && inputs[5].dtype()? == DType::Float32;
        let layout = if eligible {
            layout
        } else {
            Layout::Native
        };
        let kernel = match layout {
            Layout::Native => &self.native,
            Layout::Explicit => &self.explicit,
            Layout::Packed => &self.packed,
        };
        let packed = layout == Layout::Packed;
        Ok(kernel.dispatch(
            stream,
            inputs,
            &[
                OutputSpec::new(inputs[2].shape()?, inputs[0].dtype()?),
                OutputSpec::new(inputs[5].shape()?, inputs[5].dtype()?),
            ],
            &Dispatch::new(
                [
                    32,
                    v[3] / if packed {
                        8
                    } else {
                        1
                    },
                    v[0] * v[2],
                ],
                [
                    32,
                    if packed {
                        2
                    } else {
                        4
                    },
                    1,
                ],
            )
            .templates([
                TemplateArg::dtype("InT", inputs[0].dtype()?),
                TemplateArg::dtype("StT", inputs[5].dtype()?),
                template("DK", q[3])?,
                template("DV", v[3])?,
                template("HK", q[2])?,
                template("HV", v[2])?,
                template("STEPS", v[1])?,
            ]),
        )?)
    }
}
