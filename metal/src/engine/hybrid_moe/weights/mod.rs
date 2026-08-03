use crate::engine::{
    Array, NormWeight, Result, Stream,
    binding::BoundLinear,
    fused_gate_up::{split_interleaved_last, split_last},
};

mod load;

#[derive(Debug)]
pub(super) struct AttentionWeights {
    pub(super) query: BoundLinear,
    pub(super) key: BoundLinear,
    pub(super) value: Option<BoundLinear>,
    pub(super) output: BoundLinear,
    pub(super) query_norm: NormWeight,
    pub(super) key_norm: NormWeight,
    pub(super) rope_frequencies: Option<Array>,
}

#[derive(Debug)]
pub(super) struct DenseWeights {
    pub(super) gate: BoundLinear,
    pub(super) up: BoundLinear,
    pub(super) down: BoundLinear,
}

#[derive(Debug)]
pub(super) struct RouterWeights {
    pub(super) projection: BoundLinear,
    pub(super) norm_scale: Array,
    pub(super) expert_scale: Array,
}

#[derive(Debug)]
pub(super) struct ExpertWeights {
    pub(super) gate_up: ExpertGateUpWeights,
    pub(super) down: BoundLinear,
}

#[derive(Debug)]
pub(super) enum ExpertGateUpWeights {
    Separate {
        gate: BoundLinear,
        up: BoundLinear,
    },
    Fused {
        projection: BoundLinear,
        width: usize,
        interleaved: bool,
    },
}

impl ExpertWeights {
    pub(super) fn gather_gate_up(
        &self,
        input: &Array,
        indices: &Array,
        sorted: bool,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        match &self.gate_up {
            ExpertGateUpWeights::Separate { gate, up } => Ok((
                gate.gather(input, indices, sorted, stream)?,
                up.gather(input, indices, sorted, stream)?,
            )),
            ExpertGateUpWeights::Fused { projection, width, interleaved } => {
                let output = projection.gather(input, indices, sorted, stream)?;
                if *interleaved {
                    split_interleaved_last(&output, *width, stream)
                } else {
                    split_last(&output, *width, stream)
                }
            },
        }
    }

    pub(super) fn gather_gate_up_native(
        &self,
        input: &Array,
        indices: &Array,
        stream: &Stream,
    ) -> Result<(Array, Array)> {
        match &self.gate_up {
            ExpertGateUpWeights::Separate { gate, up } => Ok((
                gate.gather_native(input, indices, false, stream)?,
                up.gather_native(input, indices, false, stream)?,
            )),
            ExpertGateUpWeights::Fused { .. } => self.gather_gate_up(input, indices, false, stream),
        }
    }

    pub(super) const fn separate(&self) -> Option<(&BoundLinear, &BoundLinear)> {
        match &self.gate_up {
            ExpertGateUpWeights::Separate { gate, up } => Some((gate, up)),
            ExpertGateUpWeights::Fused { .. } => None,
        }
    }

    pub(super) fn tuning_format(&self) -> (i32, i32) {
        match &self.gate_up {
            ExpertGateUpWeights::Separate { gate, .. } => gate.tuning_format(),
            ExpertGateUpWeights::Fused { projection, .. } => projection.tuning_format(),
        }
    }
}

#[derive(Debug)]
pub(super) struct LayerWeights {
    pub(super) input_norm: NormWeight,
    pub(super) post_attention_norm: NormWeight,
    pub(super) pre_dense_norm: NormWeight,
    pub(super) post_dense_norm: NormWeight,
    pub(super) pre_expert_norm: NormWeight,
    pub(super) post_expert_norm: NormWeight,
    pub(super) post_feed_forward_norm: NormWeight,
    pub(super) layer_scalar: Array,
    pub(super) attention: AttentionWeights,
    pub(super) dense: DenseWeights,
    pub(super) router: RouterWeights,
    pub(super) experts: ExpertWeights,
}
