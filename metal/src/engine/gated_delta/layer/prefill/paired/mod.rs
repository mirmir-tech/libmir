mod sample;
mod validation;

use std::io::Write;

use sample::{Outcome, Variant};

use super::{Array, GatedDeltaLayer, GatedDeltaState, Result, StateArray, Stream};

#[derive(Clone, Copy)]
enum Layout {
    Rows,
    SharedParent,
}

impl GatedDeltaLayer {
    /// Test-only replay of one initialized layer. Each trial starts from an
    /// immutable snapshot; the model's live state is never advanced here.
    pub(crate) fn compare_packed_prefill(
        &self,
        input: &Array,
        states: &[&mut GatedDeltaState],
        stream: &Stream,
    ) -> Result<()> {
        assert_eq!(input.shape()?[1], 512);
        assert_eq!(states.len(), 5);
        assert!(states.iter().all(|state| state.offset == 512));
        let mut roots = vec![input];
        roots.extend(states.iter().flat_map(|state| state.graph_roots()));
        stream.eval_many(&roots)?;
        stream.synchronize()?;
        let original = states.iter().map(|state| state.snapshot()).collect::<Result<Vec<_>>>()?;
        let before = validation::StateValues::read(&original, stream)?;
        let mut exact = true;
        for layout in [Layout::Rows, Layout::SharedParent] {
            let baseline = layout.prepare(&original, stream)?;
            for sequence in [128, 512] {
                let shape = input.shape()?;
                let input = input.slice(
                    &[0, 0, 0],
                    &[states.len(), sequence, usize::try_from(shape[2])?],
                    stream,
                )?;
                stream.eval_many(&[&input])?;
                stream.synchronize()?;
                exact &= self.compare_case(&input, &baseline, layout, stream)?;
            }
        }
        let after = validation::StateValues::read(&original, stream)?;
        assert_eq!(before, after, "replays must preserve the incoming state");
        assert!(exact, "packed GDN must preserve output, recurrent state and history bitwise");
        Ok(())
    }

    fn compare_case(
        &self,
        input: &Array,
        baseline: &[GatedDeltaState],
        layout: Layout,
        stream: &Stream,
    ) -> Result<bool> {
        let rows = self.sample_prefill(Variant::Rows, input, baseline, stream)?;
        let packed = self.sample_prefill(Variant::Packed, input, baseline, stream)?;
        let exact = validation::compare(&rows, &packed, layout.name(), stream)?;
        drop((rows, packed));
        if !exact {
            return Ok(false);
        }
        // Exercise both graphs and allocation paths before any measured block.
        for _ in 0..4 {
            for variant in [Variant::Rows, Variant::Packed] {
                drop(self.sample_prefill(variant, input, baseline, stream)?);
            }
        }
        for block in 0..3 {
            for (slot, variant) in [Variant::Rows, Variant::Packed, Variant::Packed, Variant::Rows]
                .into_iter()
                .enumerate()
            {
                let result = self.sample_prefill(variant, input, baseline, stream)?;
                report(&result, layout, variant, block, slot)?;
            }
        }
        Ok(true)
    }
}

impl Layout {
    const fn name(self) -> &'static str {
        match self {
            Self::Rows => "separate_rows",
            Self::SharedParent => "shared_parent",
        }
    }

    fn prepare(
        self,
        original: &[GatedDeltaState],
        stream: &Stream,
    ) -> Result<Vec<GatedDeltaState>> {
        let mut states =
            original.iter().map(GatedDeltaState::snapshot).collect::<Result<Vec<_>>>()?;
        if matches!(self, Self::SharedParent) {
            let values = original
                .iter()
                .map(|s| required(s.value.as_ref()))
                .collect::<Result<Vec<_>>>()?;
            let histories = original
                .iter()
                .map(|s| required(s.convolution.as_ref()))
                .collect::<Result<Vec<_>>>()?;
            let values =
                StateArray::split(StateArray::join(&values, stream)?, states.len(), stream)?;
            let histories =
                StateArray::split(StateArray::join(&histories, stream)?, states.len(), stream)?;
            for ((state, value), history) in states.iter_mut().zip(values).zip(histories) {
                state.value = Some(value);
                state.convolution = Some(history);
            }
        }
        let roots = states.iter().flat_map(GatedDeltaState::graph_roots).collect::<Vec<_>>();
        stream.eval_many(&roots)?;
        stream.synchronize()?;
        Ok(states)
    }
}

fn required<T>(value: Option<T>) -> Result<T> {
    value.ok_or_else(|| super::Error::InvalidModel("GDN replay requires initialized state".into()))
}

fn report(
    result: &Outcome,
    layout: Layout,
    variant: Variant,
    block: usize,
    slot: usize,
) -> Result<()> {
    writeln!(
        std::io::stderr().lock(),
        "gdn.paired: {}",
        serde_json::json!({
            "shape": result.output.shape()?, "offset": 512,
            "layout": layout.name(), "variant": variant.name(), "block": block, "slot": slot,
            "graph_ms": result.graph_ms, "total_ms": result.total_ms,
        })
    )?;
    Ok(())
}
