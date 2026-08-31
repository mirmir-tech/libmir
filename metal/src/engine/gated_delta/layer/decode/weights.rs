use mirtal::Array;

use super::GatedDeltaLayer;
use crate::engine::{Result, binding::GraphLinear};

pub(super) struct Weights {
    pub(super) qkv: GraphLinear,
    pub(super) gate: GraphLinear,
    pub(super) beta: GraphLinear,
    pub(super) alpha: GraphLinear,
    pub(super) output: GraphLinear,
    pub(super) convolution: Array,
    pub(super) norm: Array,
    pub(super) a_log: Array,
    pub(super) dt_bias: Array,
}

impl Weights {
    pub(super) fn new(layer: &GatedDeltaLayer) -> Result<Option<Self>> {
        let Some(qkv) = GraphLinear::new(&layer.in_proj_qkv)? else {
            return Ok(None);
        };
        let Some(gate) = GraphLinear::new(&layer.in_proj_z)? else {
            return Ok(None);
        };
        let Some(beta) = GraphLinear::new(&layer.in_proj_b)? else {
            return Ok(None);
        };
        let Some(alpha) = GraphLinear::new(&layer.in_proj_a)? else {
            return Ok(None);
        };
        let Some(output) = GraphLinear::new(&layer.out_proj)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            qkv,
            gate,
            beta,
            alpha,
            output,
            convolution: layer.conv_weight.native().clone(),
            norm: layer.norm_weight.native_clone(),
            a_log: layer.a_log.native().clone(),
            dt_bias: layer.dt_bias.native().clone(),
        }))
    }
}
