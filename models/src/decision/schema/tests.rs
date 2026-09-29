use serde_json::json;

use super::DecisionSchema;
use crate::{
    Result,
    decision::{AgentConfig, Answer},
};

fn config() -> Result<AgentConfig> {
    AgentConfig::from_json(
        r#"{"max_len": 64, "head_max_len": 32, "head_layers": 2, "temperature": [1, 1, 1]}"#,
    )
}

#[test]
fn plans_enums_booleans_and_scales() -> Result<()> {
    let schema = DecisionSchema::from_json_schema(&json!({
        "type": "object",
        "properties": {
            "addressee": {"enum": ["anna", "agent", null], "description": "Who is addressed?"},
            "open_question": {"type": "boolean"},
            "urgency": {"type": "integer", "minimum": 1, "maximum": 3},
            "wait": {"anyOf": [{"type": "boolean"}, {"type": "null"}], "description": "Asked to wait?"}
        }
    }))?;
    let questions = schema.questions();
    assert_eq!(questions[0].instruction(), "Who is addressed?");
    assert_eq!(questions[0].rendered_options(), ["anna", "agent", "null"]);
    assert_eq!(questions[1].instruction(), "Is `open_question` true?");
    assert_eq!(questions[2].rendered_options(), ["level 0: 1", "level 1: 2", "level 2: 3"]);
    assert_eq!(questions[3].instruction(), "Asked to wait?");

    let config = config()?;
    let answers = vec![
        Answer::decode(&questions[0], &[0.0, 0.0, 5.0], &config)?,
        Answer::decode(&questions[1], &[0.0, 1.0], &config)?,
        Answer::decode(&questions[2], &[0.0, 3.0, 1.0], &config)?,
        Answer::decode(&questions[3], &[2.0, 0.0], &config)?,
    ];
    let values = schema.project(&answers)?;
    assert_eq!(
        serde_json::Value::Object(values),
        json!({"addressee": null, "open_question": true, "urgency": 2, "wait": false})
    );
    Ok(())
}

#[test]
fn rejects_what_a_fixed_option_set_cannot_express() {
    for property in [
        json!({"type": "string"}),
        json!({"type": "array"}),
        json!({"type": "integer", "minimum": 0}),
        json!({"type": "integer", "minimum": 0, "maximum": 10}),
        json!({"enum": []}),
        json!({"enum": ["1", 1]}),
        json!({"anyOf": [{"type": "boolean"}, {"type": "integer"}]}),
    ] {
        let schema = json!({"type": "object", "properties": {"field": property}});
        assert!(DecisionSchema::from_json_schema(&schema).is_err(), "{property}");
    }
    assert!(
        DecisionSchema::from_json_schema(&json!({"type": "object", "properties": {}})).is_err()
    );
}
