use super::*;
use crate::native::session::SessionState;

pub(super) struct Cohort {
    pub plan: DecodeReservation,
    pub history: crate::config::HistoryBatching,
    pub inputs: Vec<DecodeInput>,
    pub schedule: Vec<Vec<(usize, u64)>>,
    states: Vec<(Uuid, SessionState)>,
}

impl Cohort {
    pub fn prepare(model: &mut LoadedModel, plan: DecodeReservation) -> Result<Self> {
        Self::prepare_budget(model, plan, 3, 256)
    }

    pub fn prepare_budget(
        model: &mut LoadedModel,
        plan: DecodeReservation,
        width: usize,
        budget: usize,
    ) -> Result<Self> {
        model.stream.set_decode_reservation(plan);
        let mut cohort = None;
        let observation = run_budget_with_decode(
            model,
            MoePrefill::Default,
            2049,
            width,
            std::num::NonZeroUsize::new(budget),
            |model, inputs| {
                assert_eq!(model.sessions.len(), inputs.len());
                cohort = Some(Self {
                    plan,
                    history: crate::config::HistoryBatching::Joined,
                    inputs: inputs.to_vec(),
                    schedule: Vec::new(),
                    states: model.sessions.drain().collect(),
                });
                Ok(Observation {
                    tokens: inputs.iter().map(|input| vec![input.token]).collect(),
                    elapsed_ms: 0.0,
                })
            },
        )?;
        writeln!(
            std::io::stderr().lock(),
            "resident.prepare: {}",
            json!({"plan":plan,"width":width,"generation_budget":budget,"observation":observation})
        )?;
        let mut cohort =
            cohort.ok_or(crate::engine::Error::NullHandle("prepared benchmark cohort"))?;
        cohort.schedule = observation.schedule;
        Ok(cohort)
    }

    pub fn positions(&self) -> Result<Vec<usize>> {
        self.inputs
            .iter()
            .map(|input| {
                self.states
                    .iter()
                    .find(|(id, _)| *id == input.session)
                    .map(|(_, state)| state.position)
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| crate::engine::Error::NullHandle("resident cohort session").into())
    }

    pub fn run(&mut self, model: &mut LoadedModel, inspect: bool) -> Result<Observation> {
        assert!(model.sessions.is_empty());
        model.stream.set_decode_reservation(self.plan);
        model.stream.set_history_batching(self.history);
        model.sessions.extend(self.states.drain(..));
        assert!(model.can_decode_batch(&self.inputs), "cohort must use the resident pipeline");
        let result = if inspect {
            let mut views = Vec::new();
            let result = capture(model, &mut self.inputs, &mut views)?;
            let aliases = views
                .iter()
                .map(View::aliases_arena)
                .collect::<crate::engine::Result<Vec<_>>>()?;
            writeln!(
                std::io::stderr().lock(),
                "resident.layout: {}",
                json!({"plan":self.plan,"aliases":aliases})
            )?;
            assert_eq!(aliases.len(), 30);
            assert!(
                aliases
                    .iter()
                    .all(|&alias| alias == (self.plan == DecodeReservation::GenerationBudget))
            );
            result
        } else {
            plain(model, &mut self.inputs)?
        };
        // Both execution helpers drain pending device work before moving ownership.
        self.states.extend(model.sessions.drain());
        Ok(result)
    }
}
