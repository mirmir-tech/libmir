use llguidance::Matcher;
use serde_json::Value;

use super::{GenerationRequest, PreparedGeneration, grammar, invalid, matcher, phase::Phase};
use crate::{Model, Result, generation::constraints::validation};

/// Grammar, phase and validator for content constrained by a JSON schema.
pub(super) fn prepare(
    model: &Model,
    request: &GenerationRequest,
    prepared: &PreparedGeneration,
    schema: &Value,
) -> Result<(Matcher, Phase, jsonschema::Validator)> {
    let phase = Phase::prepare_content(model, request, prepared)?;
    let validator =
        jsonschema::validator_for(schema).map_err(|error| invalid(error.to_string()))?;
    let grammar = grammar::content(&validation::decoding_schema(schema.clone()))?;
    Ok((matcher(model, grammar)?, phase, validator))
}
