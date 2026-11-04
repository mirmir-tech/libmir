use libmir::decision::{ChoiceOption, DecisionState, Question, Verdicts};
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
pub struct Definition {
    #[serde(rename = "type")]
    kind: String,
    instructions: String,
    #[serde(default)]
    criteria: Option<Criteria>,
    #[serde(default)]
    labels: Option<Map<String, Value>>,
}

/// Criteria as `reference.py` writes them: score levels or ordered
/// `[label, description]` pairs, or an object keyed by label in written order.
#[derive(Deserialize)]
#[serde(untagged)]
enum Criteria {
    List(Vec<Value>),
    Keyed(Keyed),
}

/// Object entries in document order, which a `serde_json::Map` would sort.
struct Keyed(Vec<(String, Option<String>)>);

impl<'de> Deserialize<'de> for Keyed {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Keyed;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an object of label descriptions")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Keyed, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry()? {
                    entries.push(entry);
                }
                Ok(Keyed(entries))
            }
        }
        deserializer.deserialize_map(Visitor)
    }
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
        self.definition.question()
    }
}

impl Definition {
    pub fn question(&self) -> Result<Question, Failure> {
        let definition = self;
        let instruction = definition.instructions.clone();
        let criteria = definition.criteria.as_ref();
        Ok(match definition.kind.as_str() {
            "choice" => {
                let options = match criteria {
                    Some(Criteria::Keyed(Keyed(entries))) => entries
                        .iter()
                        .map(|(label, description)| ChoiceOption::new(label, description.clone()))
                        .collect(),
                    Some(Criteria::List(pairs)) => pairs
                        .iter()
                        .map(|pair| {
                            let label = pair
                                .get(0)
                                .and_then(Value::as_str)
                                .ok_or("choice label must be text")?;
                            let description =
                                pair.get(1).and_then(Value::as_str).map(str::to_owned);
                            Ok(ChoiceOption::new(label, description))
                        })
                        .collect::<Result<_, Failure>>()?,
                    None => return Err("choice criteria are missing".into()),
                };
                Question::choice(instruction, options)?
            },
            "score" => {
                let Some(Criteria::List(levels)) = criteria else {
                    return Err("score criteria must be a list".into());
                };
                let levels = levels
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
    criteria: Option<&Criteria>,
    labels: Option<&Map<String, Value>>,
) -> Result<Verdicts, Failure> {
    let text = |key: &str| match criteria {
        Some(Criteria::Keyed(Keyed(entries))) => entries
            .iter()
            .find(|(label, _)| label == key)
            .and_then(|(_, text)| text.clone()),
        Some(Criteria::List(_)) | None => None,
    };
    let mut verdicts = Verdicts::default().with_descriptions(text("false"), text("true"));
    if let Some(labels) = labels {
        let label = |key: &str| {
            labels.get(key).and_then(Value::as_str).ok_or("noul labels need false and true")
        };
        verdicts = verdicts.with_labels(label("false")?, label("true")?)?;
    }
    Ok(verdicts)
}
