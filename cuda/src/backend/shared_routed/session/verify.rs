//! The decoder pass of a speculative step: a captured graph over the fed
//! token and its drafts where the layers allow it, a scalar pass otherwise.

use runtime::kv::BlockTable;
use uuid::Uuid;

use super::{
    super::{
        SharedRoutedLayerTemplate, batch::CudaSharedRoutedVerifyBatch,
        plan::SharedRoutedExecutionPlan, position::text_positions,
    },
    CudaSharedRoutedModelSession,
    pass::{Drafting, Pass},
};
use crate::{Error, Result};

/// How a verification pass ran, and so how its recurrent state is kept.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VerifyPass {
    /// In the plan's verification graph, which holds the recurrent state.
    Graphed,
    /// As a scalar pass with Mamba-2 snapshots armed in the session.
    Scalar,
}

impl CudaSharedRoutedModelSession {
    /// Feeds `token` and the first `rows - 1` device drafts through the
    /// decoder; `table` holds the session through the last draft. The
    /// prediction plan of `rows` rows then holds every normalized row.
    pub(super) fn verify_pass(
        &mut self,
        session_id: Uuid,
        token: u32,
        rows: usize,
        table: &BlockTable,
    ) -> Result<VerifyPass> {
        if rows < 2 || self.position_delta != 0 || !self.graphs_verification() {
            let mut tokens = vec![0; rows];
            tokens[0] = token;
            // Any later failure retires the session; a partially executed
            // pass cannot safely be retried by merely clearing the arm flag.
            self.arm_rollback(rows - 1)?;
            let positions = text_positions(self.position, rows, self.position_delta)?;
            self.prefill_inner(Pass {
                drafting: Drafting::Rows,
                device_drafts: rows - 1,
                ..Pass::text(session_id, &tokens, &positions, table)
            })?;
            return Ok(VerifyPass::Scalar);
        }
        let plans = self.template.plans.clone();
        let mut plans = plans
            .lock()
            .map_err(|_| Error::State("shared-routed execution plan cache is poisoned".into()))?;
        if plans.get_mut(rows).is_none() {
            let plan = SharedRoutedExecutionPlan::new(&self.template, rows)?;
            plans.insert(rows, plan);
        }
        let plan = plans
            .get_mut(rows)
            .ok_or(Error::InvalidDecoderKernel("verification plan is missing"))?;
        if plan.verify.is_none() {
            plan.verify = Some(CudaSharedRoutedVerifyBatch::new(&self.template, rows)?);
        }
        let (Some(verify), Some(prediction)) = (plan.verify.as_mut(), plan.prediction.as_mut())
        else {
            return Err(Error::InvalidDecoderKernel("verification needs the prediction module"));
        };
        let block_count = self.template.cache_config().block_count;
        verify.execute(self, token, table, block_count)?;
        let normalized = verify.normalized()?;
        self.template.backend.inner.stream.copy_device_range(
            normalized,
            0..normalized.len(),
            &mut prediction.hidden,
            0,
        )?;
        drop(plans);
        self.position += rows;
        Ok(VerifyPass::Graphed)
    }

    /// Keeps the state after `kept` tokens of the graphed pass of `rows`
    /// rows that started at `start`.
    pub(super) fn commit_verified(&mut self, rows: usize, start: usize, kept: usize) -> Result<()> {
        let plans = self.template.plans.clone();
        let mut plans = plans
            .lock()
            .map_err(|_| Error::State("shared-routed execution plan cache is poisoned".into()))?;
        plans
            .get_mut(rows)
            .and_then(|plan| plan.verify.as_mut())
            .ok_or(Error::InvalidDecoderKernel("verification graph is missing"))?
            .commit(self, kept)?;
        drop(plans);
        self.position = start + kept;
        Ok(())
    }

    /// Lone verification passes run as a graph when every layer is Mamba-2
    /// or attention.
    fn graphs_verification(&self) -> bool {
        self.template.layers.iter().all(|layer| {
            matches!(
                layer,
                SharedRoutedLayerTemplate::Mamba2(_) | SharedRoutedLayerTemplate::Attention(_)
            )
        })
    }
}
