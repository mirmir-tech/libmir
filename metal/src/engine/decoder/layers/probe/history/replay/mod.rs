use std::io::Write;

use crate::engine::{Array, Error, Result, Stream};
mod cost;

pub struct Replay {
    pub layer: usize,
    query: Array,
    keys: Vec<Array>,
    values: Vec<Array>,
    joined: [Array; 2],
    output: Array,
    scale: f32,
    causal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
enum Plan {
    Assemble,
    Resident,
}

impl Replay {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        layer: usize,
        query: &Array,
        keys: &[&Array],
        values: &[&Array],
        joined: [&Array; 2],
        output: &Array,
        scale: f32,
        causal: bool,
    ) -> Result<Self> {
        Ok(Self {
            layer,
            query: query.snapshot()?,
            keys: keys.iter().map(|x| x.snapshot()).collect::<Result<_>>()?,
            values: values.iter().map(|x| x.snapshot()).collect::<Result<_>>()?,
            joined: [joined[0].snapshot()?, joined[1].snapshot()?],
            output: output.snapshot()?,
            scale,
            causal,
        })
    }

    pub fn roots(&self) -> Vec<&Array> {
        let mut roots = vec![&self.query, &self.output, &self.joined[0], &self.joined[1]];
        roots.extend(self.keys.iter().chain(&self.values));
        roots
    }

    fn forward(&self, plan: Plan, stream: &Stream) -> Result<Array> {
        match plan {
            Plan::Assemble => self.query.scaled_dot_product_attention(
                &Array::concatenate(&self.keys.iter().collect::<Vec<_>>(), 0, stream)?,
                &Array::concatenate(&self.values.iter().collect::<Vec<_>>(), 0, stream)?,
                self.scale,
                self.causal,
                stream,
            ),
            Plan::Resident => self.query.scaled_dot_product_attention(
                &self.joined[0], &self.joined[1], self.scale, self.causal, stream,
            ),
        }
    }

    pub fn inspect(&self, stream: &Stream) -> Result<()> {
        stream.eval_many(&self.roots())?;
        stream.synchronize()?;
        let mut buffers = Vec::new();
        for (kind, joined, inputs) in
            [("key", &self.joined[0], &self.keys), ("value", &self.joined[1], &self.values)]
        {
            let allocation = joined
                .native()
                .allocation()?
                .ok_or(Error::NullHandle("joined K/V allocation"))?;
            let mut aliases = Vec::new();
            let mut input_bytes = Vec::new();
            for input in inputs {
                let source = input
                    .native()
                    .allocation()?
                    .ok_or(Error::NullHandle("source K/V allocation"))?;
                aliases.push(allocation == source);
                input_bytes.push(source.bytes());
            }
            assert!(
                aliases.iter().all(|&alias| !alias),
                "joined storage unexpectedly shares input"
            );
            buffers.push(serde_json::json!({"kind":kind,"shape":joined.shape()?,"logical_bytes":joined.byte_len()?,
                "allocated_bytes":allocation.bytes(),"aliases_input":aliases,"input_allocation_bytes":input_bytes}));
        }
        for plan in [Plan::Assemble, Plan::Resident] {
            let output = self.forward(plan, stream)?;
            let actual = output.to_vec_f32(stream)?;
            let expected = self.output.to_vec_f32(stream)?;
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits()),
                "history replay changed attention output"
            );
        }
        writeln!(
            std::io::stderr().lock(),
            "history.storage: {}",
            serde_json::json!({
                "layer":self.layer,"query_shape":self.query.shape()?,"buffers":buffers,"replay_bitwise_equal":true,
            })
        )?;
        Ok(())
    }
}
