mod field;

use serde_json::{Map, Value};

use super::{Answer, Question, Verdict};
use crate::error::{ModelsError, Result};

const MAX_PROPERTIES: usize = 32;

/// How a decided answer becomes a schema value.
#[derive(Debug, Clone, PartialEq)]
enum Projection {
    /// The enum value of the chosen option, in option order.
    Choice(Vec<Value>),
    /// `minimum` plus the most probable level.
    Score {
        minimum: i64,
    },
    Boolean,
}

/// One schema property asked as a question.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaField {
    pub name: String,
    pub question: Question,
    projection: Projection,
}

/// A flat JSON schema whose properties are enums, booleans, or bounded
/// integer scales, planned as one question per property.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionSchema {
    fields: Vec<SchemaField>,
}

impl DecisionSchema {
    pub fn from_json_schema(schema: &Value) -> Result<Self> {
        let object = schema.as_object().ok_or_else(|| invalid("expected a JSON schema object"))?;
        if object.get("type").is_some_and(|kind| kind != "object") {
            return Err(invalid("the top level must be an object with properties"));
        }
        let properties = object
            .get("properties")
            .and_then(Value::as_object)
            .filter(|properties| !properties.is_empty())
            .ok_or_else(|| invalid("'properties' must be a non-empty object"))?;
        if properties.len() > MAX_PROPERTIES {
            return Err(invalid(format!(
                "{} properties exceed the limit of {MAX_PROPERTIES}",
                properties.len()
            )));
        }
        let fields = properties
            .iter()
            .map(|(name, property)| {
                field::plan(&format!("properties.{name}"), name, property, None)
            })
            .collect::<Result<_>>()?;
        Ok(Self { fields })
    }

    #[must_use]
    pub fn fields(&self) -> &[SchemaField] {
        &self.fields
    }

    /// Questions in field order, ready to be asked of one state.
    #[must_use]
    pub fn questions(&self) -> Vec<Question> {
        self.fields.iter().map(|field| field.question.clone()).collect()
    }

    /// Projects answers, one per field in field order, onto schema values.
    pub fn project(&self, answers: &[Answer]) -> Result<Map<String, Value>> {
        if answers.len() != self.fields.len() {
            return Err(invalid(format!(
                "{} answers for {} schema fields",
                answers.len(),
                self.fields.len()
            )));
        }
        self.fields
            .iter()
            .zip(answers)
            .map(|(field, answer)| Ok((field.name.clone(), project(field, answer)?)))
            .collect()
    }
}

fn project(field: &SchemaField, answer: &Answer) -> Result<Value> {
    let best = answer
        .probabilities
        .iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |best, (index, &p)| {
            if p > best.1 {
                (index, p)
            } else {
                best
            }
        })
        .0;
    match (&field.projection, &answer.verdict) {
        (Projection::Choice(values), Verdict::Choice { .. }) => Ok(values[best].clone()),
        (Projection::Score { minimum }, Verdict::Score { .. }) => {
            Ok(Value::from(minimum + i64::try_from(best)?))
        },
        (Projection::Boolean, Verdict::YesNo { probability }) => {
            Ok(Value::Bool(*probability >= 0.5))
        },
        (Projection::Choice(_) | Projection::Score { .. } | Projection::Boolean, _) => {
            Err(invalid(format!("answer kind does not match schema field `{}`", field.name)))
        },
    }
}

fn invalid(message: impl Into<String>) -> ModelsError {
    ModelsError::InvalidConfig(message.into())
}

#[cfg(test)]
mod tests;
