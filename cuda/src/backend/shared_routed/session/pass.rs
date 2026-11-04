//! One scalar pass of the session over a fixed-shape execution plan.

use mircuda::{DeviceBuffer, bf16};
use runtime::kv::BlockTable;
use uuid::Uuid;

use super::{super::plan::SharedRoutedExecutionPlan, CudaSharedRoutedModelSession, bf16_tensor};
use crate::{Error, Result};

/// What a pass leaves for the prediction module.
#[derive(Clone, Copy, Debug)]
pub(super) enum Drafting<'a> {
    /// Only the last decoder row is normalized, for its logits.
    Off,
    /// Every decoder row is normalized and kept for a later prediction.
    Rows,
    /// Every row is normalized and the module runs at once, with the token
    /// after each row.
    Predict(&'a [u32]),
}

/// The inputs of one scalar pass.
#[derive(Clone, Copy)]
pub(super) struct Pass<'a> {
    pub(super) session_id: Uuid,
    pub(super) tokens: &'a [u32],
    pub(super) positions: &'a [u32],
    pub(super) table: &'a BlockTable,
    pub(super) image_span: Option<(usize, usize)>,
    pub(super) image: Option<&'a DeviceBuffer<bf16>>,
    pub(super) drafting: Drafting<'a>,
    /// Trailing tokens taken from the session's device drafts instead of
    /// `tokens`.
    pub(super) device_drafts: usize,
}

impl<'a> Pass<'a> {
    pub(super) const fn text(
        session_id: Uuid,
        tokens: &'a [u32],
        positions: &'a [u32],
        table: &'a BlockTable,
    ) -> Self {
        Self {
            session_id,
            tokens,
            positions,
            table,
            image_span: None,
            image: None,
            drafting: Drafting::Off,
            device_drafts: 0,
        }
    }
}

impl CudaSharedRoutedModelSession {
    pub(super) fn prefill_inner(&mut self, pass: Pass<'_>) -> Result<&DeviceBuffer<bf16>> {
        let count = pass.tokens.len();
        let plans = self.template.plans.clone();
        let mut plans = plans
            .lock()
            .map_err(|_| Error::State("shared-routed execution plan cache is poisoned".into()))?;
        if let Some(plan) = plans.get_mut(count) {
            self.execute_plan(&pass, plan)?;
        } else {
            plans.reserve(count);
            let mut plan = SharedRoutedExecutionPlan::new(&self.template, count)?;
            self.execute_plan(&pass, &mut plan)?;
            plans.insert(count, plan);
        }
        drop(plans);
        Ok(&self.logits)
    }

    fn execute_plan(
        &mut self,
        pass: &Pass<'_>,
        plan: &mut SharedRoutedExecutionPlan,
    ) -> Result<()> {
        let Pass {
            session_id,
            tokens,
            positions,
            table,
            image_span,
            image,
            drafting,
            device_drafts,
        } = *pass;
        self.validate(tokens, positions, table)?;
        for token in tokens {
            self.embedding.validate_token(*token)?;
        }
        let count = tokens.len();
        plan.upload(&self.template, tokens, positions)?;
        if device_drafts > 0 {
            let drafts = self
                .drafts
                .as_ref()
                .ok_or(Error::InvalidDecoderKernel("session has no device drafts"))?;
            plan.load_tail_tokens(&self.template, drafts, device_drafts)?;
        }
        let hidden = plan.execute(
            &self.template, &self.embedding, &mut self.states, session_id, table, self.position,
            image_span, image,
        )?;
        let stream = &self.template.backend.inner.stream;
        match drafting {
            Drafting::Off => {
                // A pass without the prediction module leaves its K/V behind.
                self.drafted = 0;
                self.select_row
                    .execute(stream, hidden, &mut self.last_hidden, count - 1, count)?;
                self.final_norm.execute(
                    stream,
                    &self.last_hidden,
                    bf16_tensor(self.template.final_norm_weight())?,
                    &mut self.normalized,
                )?;
                self.output.execute(&self.normalized, &mut self.logits)?;
            },
            Drafting::Rows => self.normalize_rows(plan)?,
            Drafting::Predict(next) => {
                self.normalize_rows(plan)?;
                self.predict_plan(plan, next, session_id, table, self.position)?;
            },
        }
        self.position = self
            .position
            .checked_add(count)
            .ok_or(Error::InvalidDecoderKernel("shared-routed session position overflow"))?;
        Ok(())
    }
}
