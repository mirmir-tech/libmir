mod budget;
mod content;
mod drafts;
mod flow;
mod grammar;
mod phase;
pub(super) mod prompt;
#[cfg(test)]
mod tests;
mod validation;

pub(super) use flow::{advance_generation, normalize, sampling, transition_stop, validate_content};
use llguidance::{Matcher, api::TopLevelGrammar};
use models::{chat::ToolCallPrefix, generation::GenerationSettings};

use super::{GenerationRequest, PreparedGeneration, ResponseFormat, ToolConstraints};
use crate::{Model, Result};

/// What the grammar constrains.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Target {
    /// The arguments of the named tool, inside its native call syntax.
    Tool,
    /// The final content, a JSON value of the response format's schema.
    Content,
}

pub(super) struct OutputConstraint {
    matcher: Matcher,
    phase: phase::Phase,
    vocab: usize,
    validator: jsonschema::Validator,
    target: Target,
}

pub(super) fn invalid(message: impl Into<String>) -> crate::Error {
    models::ModelsError::InvalidConfig(format!("tool constraints: {}", message.into())).into()
}

impl OutputConstraint {
    /// The reasoning delimiter and the tokens the budget still allows, while
    /// the model reasons unconstrained; `None` once the grammar or the budget
    /// exit takes over.
    pub(super) fn reasoning_bounds(&self) -> Option<(u32, Option<usize>)> {
        self.phase.reasoning_bounds()
    }

    pub(super) fn prepare(
        model: &Model,
        request: &GenerationRequest,
        prepared: &PreparedGeneration,
        settings: GenerationSettings,
    ) -> Result<Option<Self>> {
        budget::validate(request, settings.max_tokens)?;
        let schema = match (request.tool_constraints, &request.response_format) {
            (ToolConstraints::None, ResponseFormat::Text) => return Ok(None),
            (ToolConstraints::Schema, ResponseFormat::JsonSchema(_)) => {
                return Err(invalid("schema tools and a JSON response format are exclusive"));
            },
            (ToolConstraints::Schema, ResponseFormat::Text) => None,
            (ToolConstraints::None, ResponseFormat::JsonSchema(schema)) => Some(schema),
        };
        if !matches!(
            model.engine().target(),
            foundation::model::BackendTarget::Cuda | foundation::model::BackendTarget::Metal
        ) {
            return Err(invalid("schema-constrained device sampling requires CUDA or Metal"));
        }
        if (settings.repetition_penalty - 1.0).abs() > f32::EPSILON
            || settings.ignore_eos
            || settings.min_tokens > 0
        {
            return Err(invalid(
                "schema mode requires repetition_penalty=1, min_tokens=0 and ignore_eos=false",
            ));
        }
        let vocab = model.descriptor().tokenizer().vocab_size();
        if let Some(schema) = schema {
            let (matcher, phase, validator) = content::prepare(model, request, prepared, schema)?;
            return Ok(Some(Self {
                matcher,
                phase,
                vocab,
                validator,
                target: Target::Content,
            }));
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
        let matcher = matcher(model, grammar)?;
        Ok(Some(Self {
            matcher,
            phase,
            vocab,
            validator,
            target: Target::Tool,
        }))
    }

    /// Masks for the drafts after the token just consumed, while the grammar
    /// holds; `None` outside it.
    pub(super) fn draft_masks(
        &mut self,
        drafts: &[u32],
    ) -> Result<Option<Vec<runtime::speculation::MaskedDraft>>> {
        if !self.phase.is_grammar() {
            return Ok(None);
        }
        drafts::masks(&mut self.matcher, drafts, self.vocab).map(Some)
    }

    pub(super) fn reasoning_exit(&self, request: &GenerationRequest) -> Option<(usize, Vec<u32>)> {
        let super::ReasoningCyclePolicy::ExitReasoning { min_tokens } = request.reasoning_cycle
        else {
            return None;
        };
        self.phase.end_token().map(|end| (min_tokens, vec![end]))
    }

    fn consume(&mut self, token: u32) -> Result<()> {
        if self.phase.observe(token) {
            return Ok(());
        }
        self.matcher.consume_token(token).map_err(|error| invalid(error.to_string()))
    }

    fn complete(&mut self) -> Result<bool> {
        if !self.phase.is_grammar() {
            return Ok(false);
        }
        self.matcher.is_accepting().map_err(|error| invalid(error.to_string()))
    }
}

/// A grammar matcher over the model's tokenizer; rejects grammars llguidance
/// only partially supports.
fn matcher(model: &Model, grammar: String) -> Result<Matcher> {
    let parser = model
        .descriptor()
        .tool_parser_factory()?
        .create_parser(TopLevelGrammar::from_lark(grammar))
        .map_err(|error| invalid(error.to_string()))?;
    let mut matcher = Matcher::new(Ok(parser));
    let warnings = matcher.grammar_warnings();
    if warnings.is_empty() {
        Ok(matcher)
    } else {
        Err(invalid(format!("unsupported schema: {}", warnings.join("; "))))
    }
}
