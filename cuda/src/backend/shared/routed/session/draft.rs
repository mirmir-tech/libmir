//! Drafting passes: the decoder and the prediction module over the same rows.

use mircuda::{DeviceBuffer, bf16};
use runtime::kv::BlockTable;
use uuid::Uuid;

use super::{
    super::{plan::SharedRoutedExecutionPlan, position::text_positions},
    CudaSharedRoutedModelSession,
    pass::{Drafting, Pass},
};
use crate::{Error, Result};

/// The decoder or prediction-module output of a drafting pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DraftRow {
    Decoder(usize),
    Prediction(usize),
}

impl CudaSharedRoutedModelSession {
    /// Prefills `tokens` and runs the prediction module over the same rows,
    /// with `next_tokens[i]` the token after row `i`. The session logits are
    /// the last decoder row's; [`Self::row_logits`] reads any row.
    pub fn prefill_drafting(
        &mut self,
        session_id: Uuid,
        tokens: &[u32],
        next_tokens: &[u32],
        table: &BlockTable,
    ) -> Result<&DeviceBuffer<bf16>> {
        if next_tokens.len() != tokens.len() {
            return Err(Error::InvalidDecoderKernel("draft tokens differ from prompt rows"));
        }
        let positions = text_positions(self.position, tokens.len(), self.position_delta)?;
        self.prefill_inner(Pass {
            drafting: Drafting::Predict(next_tokens),
            ..Pass::text(session_id, tokens, &positions, table)
        })
    }

    /// Prefills `tokens` and keeps every normalized decoder row for a later
    /// [`Self::predict`]; the session logits are the last row's.
    pub fn prefill_rows(
        &mut self,
        session_id: Uuid,
        tokens: &[u32],
        table: &BlockTable,
    ) -> Result<&DeviceBuffer<bf16>> {
        let positions = text_positions(self.position, tokens.len(), self.position_delta)?;
        self.prefill_inner(Pass {
            drafting: Drafting::Rows,
            ..Pass::text(session_id, tokens, &positions, table)
        })
    }

    /// Runs the prediction module over the rows of the last pass, which had
    /// `tokens` rows ending at the session position.
    // The plan guard is used by the last statement; the lint cannot see it.
    #[allow(clippy::significant_drop_tightening)]
    pub fn predict(
        &mut self,
        session_id: Uuid,
        tokens: usize,
        next_tokens: &[u32],
        table: &BlockTable,
    ) -> Result<()> {
        let start = self
            .position
            .checked_sub(tokens)
            .ok_or(Error::InvalidDecoderKernel("prediction rows precede the session"))?;
        let plans = self.template.plans.clone();
        let mut plans = plans
            .lock()
            .map_err(|_| Error::State("shared-routed execution plan cache is poisoned".into()))?;
        let plan = plans
            .get_mut(tokens)
            .ok_or(Error::InvalidDecoderKernel("no drafting pass of this shape"))?;
        self.predict_plan(plan, next_tokens, session_id, table, start)
    }

    /// Logits of one row of the last drafting pass, which had `tokens` rows.
    pub fn row_logits(&mut self, tokens: usize, row: DraftRow) -> Result<&DeviceBuffer<bf16>> {
        let plans = self.template.plans.clone();
        let mut plans = plans
            .lock()
            .map_err(|_| Error::State("shared-routed execution plan cache is poisoned".into()))?;
        let prediction = plans
            .get_mut(tokens)
            .and_then(|plan| plan.prediction.as_ref())
            .ok_or(Error::InvalidDecoderKernel("no drafting pass of this shape"))?;
        let (rows, index) = match row {
            DraftRow::Decoder(index) => (&prediction.hidden, index),
            DraftRow::Prediction(index) => (&prediction.drafted, index),
        };
        self.select_row.execute(
            &self.template.backend.inner.stream,
            rows,
            &mut self.normalized,
            index,
            tokens,
        )?;
        drop(plans);
        self.output.execute(&self.normalized, &mut self.logits)?;
        Ok(&self.logits)
    }

    /// Normalizes every decoder row of the pass and takes the last row's
    /// logits.
    pub(super) fn normalize_rows(&mut self, plan: &mut SharedRoutedExecutionPlan) -> Result<()> {
        let (hidden, prediction) = plan.output_and_prediction();
        let prediction = prediction
            .ok_or(Error::InvalidDecoderKernel("model has no prediction module loaded"))?;
        prediction.normalize_decoder(hidden, &self.template.final_norm)?;
        let count = prediction.tokens();
        self.select_row.execute(
            &self.template.backend.inner.stream,
            &prediction.hidden,
            &mut self.normalized,
            count - 1,
            count,
        )?;
        self.output.execute(&self.normalized, &mut self.logits)
    }

    /// Runs the prediction module over the plan's normalized rows at
    /// `start..start + tokens`.
    pub(super) fn predict_plan(
        &mut self,
        plan: &mut SharedRoutedExecutionPlan,
        next_tokens: &[u32],
        session_id: Uuid,
        table: &BlockTable,
        start: usize,
    ) -> Result<()> {
        let prediction = plan
            .prediction
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("model has no prediction module loaded"))?;
        let state = self
            .prediction
            .as_mut()
            .ok_or(Error::InvalidDecoderKernel("session has no prediction module state"))?;
        for token in next_tokens {
            self.embedding.validate_token(*token)?;
        }
        prediction.upload_tokens(next_tokens)?;
        prediction.execute(&self.embedding, state, session_id, table, start)
    }
}
