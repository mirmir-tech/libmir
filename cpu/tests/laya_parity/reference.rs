use models::decision::{ChoiceOption, DecisionState, Question, Verdicts};
use serde::Deserialize;
use serde_json::{Map, Value};

/// Rows and fp32 CPU logits produced by the upstream Python runtime; see
/// Workmir `benchmarks/2026-09-29-laya-reference`.
#[derive(Deserialize)]
pub struct Reference {
    pub cases: Vec<Case>,
}

#[derive(Deserialize)]
pub struct Case {
    #[serde(rename = "case")]
    pub name: String,
    pub question: String,
    state: StateRecord,
    pub serialized_state: String,
    definition: Definition,
    pub tokens: Vec<u32>,
    pub markers: Vec<usize>,
    pub logits: Vec<f32>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
enum StateRecord {
    Text(String),
    Document(Value),
    Conversation(Vec<Value>),
}

#[derive(Deserialize)]
struct Definition {
    #[serde(rename = "type")]
    kind: String,
    instructions: String,
    #[serde(default)]
    criteria: Option<Value>,
    #[serde(default)]
    labels: Option<Map<String, Value>>,
}

pub type Failure = Box<dyn std::error::Error>;

impl Case {
    pub fn state(&self) -> Result<DecisionState, Failure> {
        Ok(match &self.state {
            StateRecord::Text(text) => DecisionState::text(text.clone()),
            StateRecord::Document(value) => DecisionState::document(value)?,
            StateRecord::Conversation(turns) => DecisionState::conversation(turns)?,
        })
    }

    pub fn question(&self) -> Result<Question, Failure> {
        let definition = &self.definition;
        let instruction = definition.instructions.clone();
        let criteria = definition.criteria.as_ref();
        Ok(match definition.kind.as_str() {
            "choice" => {
                let options = criteria
                    .and_then(Value::as_array)
                    .ok_or("choice criteria must be ordered [label, description] pairs")?
                    .iter()
                    .map(|pair| {
                        let label = pair
                            .get(0)
                            .and_then(Value::as_str)
                            .ok_or("choice label must be text")?;
                        let description = pair.get(1).and_then(Value::as_str).map(str::to_owned);
                        Ok(ChoiceOption::new(label, description))
                    })
                    .collect::<Result<_, Failure>>()?;
                Question::choice(instruction, options)?
            },
            "score" => {
                let levels = criteria
                    .and_then(Value::as_array)
                    .ok_or("score criteria must be a list")?
                    .iter()
                    .map(|level| {
                        level.as_str().map(str::to_owned).ok_or("score levels must be text")
                    })
                    .collect::<Result<_, _>>()?;
                Question::score(instruction, levels)?
            },
            "noul" => {
                Question::yes_no(instruction, verdicts(criteria, definition.labels.as_ref())?)?
            },
            other => return Err(format!("unknown question type {other}").into()),
        })
    }
}

fn verdicts(
    criteria: Option<&Value>,
    labels: Option<&Map<String, Value>>,
) -> Result<Verdicts, Failure> {
    let text = |map: Option<&Value>, key: &str| {
        map.and_then(|map| map.get(key)).and_then(Value::as_str).map(str::to_owned)
    };
    let mut verdicts =
        Verdicts::default().with_descriptions(text(criteria, "false"), text(criteria, "true"));
    if let Some(labels) = labels {
        let label = |key: &str| {
            labels.get(key).and_then(Value::as_str).ok_or("noul labels need false and true")
        };
        verdicts = verdicts.with_labels(label("false")?, label("true")?)?;
    }
    Ok(verdicts)
}
