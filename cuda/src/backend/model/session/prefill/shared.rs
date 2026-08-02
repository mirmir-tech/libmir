use runtime::kv::{BlockTable, KvWritePlan};
use uuid::Uuid;

use super::CudaMoeModelSession;
use crate::Result;

impl CudaMoeModelSession {
    pub(super) fn execute_shared_prefill(
        &mut self,
        session_id: Uuid,
        table: &BlockTable,
        write_offset: usize,
        tokens: usize,
    ) -> Result<()> {
        let mut plans = self.packed_prefill.take(tokens).unwrap_or_default();
        let result = (|| {
            for (index, layer) in self.layers.iter_mut().enumerate() {
                let plan =
                    KvWritePlan::prefill(session_id, layer.layer(), table, write_offset, tokens)?;
                let (input, output) = if index.is_multiple_of(2) {
                    (&self.prefill_first, &mut self.prefill_second)
                } else {
                    (&self.prefill_second, &mut self.prefill_first)
                };
                let signature = layer.prefill_signature();
                let plan_index = plans.iter().position(|plan| plan.supports(signature));
                let plan_index = if let Some(index) = plan_index {
                    index
                } else {
                    plans.push(layer.instantiate_shared_prefill(tokens)?);
                    plans.len() - 1
                };
                let prefill = plans[plan_index].borrow();
                if let Some(graph) = self.decode_graph.as_mut() {
                    graph.execute_prefill(
                        index, prefill, input, &plan, table, write_offset, output,
                    )?;
                } else {
                    layer.execute_shared_prefill(
                        prefill, input, output, &plan, table, write_offset,
                    )?;
                }
            }
            Ok(())
        })();
        self.packed_prefill.insert(tokens, plans);
        result
    }
}
