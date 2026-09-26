mod grammar;
mod phase;
pub(super) mod prompt;
#[cfg(test)]
mod tests;
mod validation;

use llguidance::{Matcher, api::TopLevelGrammar};
use models::{chat::ToolCallPrefix, generation::GenerationSettings};
use runtime::backend::{DeviceSampling, SamplingLogits, TokenMask};

use super::{GenerationRequest, PreparedGeneration, ToolConstraints};
use crate::{Model, Result};

pub(super) struct ToolConstraint {
    matcher: Matcher,
    phase: phase::Phase,
    vocab: usize,
    validator: jsonschema::Validator,
}

pub(super) fn invalid(message: impl Into<String>) -> crate::Error {
    models::ModelsError::InvalidConfig(format!("tool constraints: {}", message.into())).into()
}

impl ToolConstraint {
    pub(super) fn prepare(
        model: &Model,
        request: &GenerationRequest,
        prepared: &PreparedGeneration,
        settings: GenerationSettings,
    ) -> Result<Option<Self>> {
        if request.tool_constraints == ToolConstraints::None {
            return Ok(None);
        }
        if model.engine().target() != foundation::model::BackendTarget::Cuda {
            return Err(invalid("schema-constrained device sampling currently requires CUDA"));
        }
        if (settings.repetition_penalty - 1.0).abs() > f32::EPSILON
            || settings.ignore_eos
            || settings.min_tokens > 0
        {
            return Err(invalid(
                "schema mode requires repetition_penalty=1, min_tokens=0 and ignore_eos=false",
            ));
        }
        let (prefix, phase) = phase::Phase::prepare(model, request, prepared)?;
        let ToolCallPrefix::XmlFunction(name) = &prefix;
        let tool = request
            .conversation
            .tools
            .iter()
            .find(|tool| tool.function.name == *name)
            .ok_or_else(|| invalid("named tool missing"))?;
        let factory = model.descriptor().tool_parser_factory()?;
        let tool_end = factory.tok_env().tok_trie().get_special_token("</tool_call>");
        let validator = jsonschema::validator_for(&tool.function.parameters)
            .map_err(|error| invalid(error.to_string()))?;
        let schema = validation::decoding_schema(tool.function.parameters.clone());
        let grammar = grammar::compile(&schema, tool_end)?;
        let grammar = if phase.is_reasoning() {
            grammar::with_prefix(&grammar, &prefix)?
        } else {
            grammar
        };
        let parser = factory
            .create_parser(TopLevelGrammar::from_lark(grammar))
            .map_err(|error| invalid(error.to_string()))?;
        let mut matcher = Matcher::new(Ok(parser));
        let warnings = matcher.grammar_warnings();
        if !warnings.is_empty() {
            return Err(invalid(format!("unsupported schema: {}", warnings.join("; "))));
        }
        Ok(Some(Self {
            matcher,
            phase,
            validator,
            vocab: model.descriptor().tokenizer().vocab_size(),
        }))
    }

    pub(super) fn is_tool(&self) -> bool {
        !self.phase.is_reasoning()
    }

    pub(super) fn reasoning_exit(&self, request: &GenerationRequest) -> Option<(usize, Vec<u32>)> {
        let super::ReasoningCyclePolicy::ExitReasoning { min_tokens } = request.reasoning_cycle
        else {
            return None;
        };
        self.phase.end_token().map(|end| (min_tokens, vec![end]))
    }

    pub(super) fn consume(&mut self, token: u32) -> Result<()> {
        if self.phase.observe(token) {
            return Ok(());
        }
        self.matcher.consume_token(token).map_err(|error| invalid(error.to_string()))
    }

    pub(super) fn complete(&mut self) -> Result<bool> {
        if self.phase.is_reasoning() {
            return Ok(false);
        }
        self.matcher.is_accepting().map_err(|error| invalid(error.to_string()))
    }
}

pub(super) fn sampling(
    constraint: &mut Option<ToolConstraint>,
    policy: SamplingLogits,
) -> Result<SamplingLogits> {
    let Some(constraint) = constraint else {
        return Ok(policy);
    };
    if constraint.phase.is_reasoning() {
        return Ok(policy);
    }
    let sampling = match policy {
        SamplingLogits::None => DeviceSampling::Greedy,
        SamplingLogits::SampleTopK { k, temperature, draw, .. } => {
            DeviceSampling::Random { top_k: k, top_p: 1.0, temperature, draw }
        },
        SamplingLogits::Sample { top_k, top_p, temperature, draw, .. } => {
            DeviceSampling::Random { top_k, top_p, temperature, draw }
        },
        _ => return Err(invalid("schema mode requires device sampling without history")),
    };
    let bits = constraint.matcher.compute_mask().map_err(|error| invalid(error.to_string()))?;
    let mask = TokenMask::new(bits.as_slice().to_vec(), constraint.vocab)?;
    Ok(SamplingLogits::Masked { mask, sampling })
}

pub(super) fn advance(constraint: &mut Option<ToolConstraint>, token: u32) -> Result<bool> {
    let Some(constraint) = constraint else {
        return Ok(false);
    };
    constraint.consume(token)?;
    constraint.complete()
}

/// Once the grammar owns the output, neither history penalties nor injected
/// reasoning exits may override its device mask.
pub(super) fn advance_generation(
    constraint: &mut Option<ToolConstraint>,
    token: u32,
    cycle: &mut super::CycleRecovery,
) -> Result<bool> {
    let complete = advance(constraint, token)?;
    if constraint.as_ref().is_some_and(ToolConstraint::is_tool) {
        cycle.disable();
    }
    Ok(complete)
}

pub(super) fn validate(constraint: Option<&mut ToolConstraint>, calls: &str) -> Result<()> {
    if let Some(constraint) = constraint {
        if !constraint.complete()? {
            return Err(crate::Error::InvalidToolCall(
                "schema-constrained tool generation ended before completion".into(),
            ));
        }
        validation::validate(&constraint.validator, calls)?;
    }
    Ok(())
}
