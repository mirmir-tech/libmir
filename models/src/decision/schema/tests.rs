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

fn score_field(minimum: i64, maximum: i64) -> Result<DecisionSchema> {
    DecisionSchema::from_json_schema(&json!({
        "type": "object",
        "properties": {"field": {"type": "integer", "minimum": minimum, "maximum": maximum}}
    }))
}

#[test]
fn rejects_score_ranges_that_overflow_or_exceed_the_level_limit() {
    for (minimum, maximum) in [
        (i64::MIN, i64::MAX),
        (i64::MIN, 0),
        (-1, i64::MAX),
        (i64::MIN, i64::MIN + 10),
        (i64::MAX - 10, i64::MAX),
        (3, 2),
        (i64::MAX, i64::MIN),
        (0, 10),
    ] {
        assert!(score_field(minimum, maximum).is_err(), "{minimum}..={maximum}");
    }
}

#[test]
fn accepts_single_and_maximum_score_ranges_at_the_extremes() -> Result<()> {
    for (minimum, maximum, levels) in [
        (5, 5, 1),
        (i64::MIN, i64::MIN, 1),
        (i64::MAX, i64::MAX, 1),
        (0, 9, 10),
        (i64::MIN, i64::MIN + 9, 10),
        (i64::MAX - 9, i64::MAX, 10),
    ] {
        let schema = score_field(minimum, maximum)?;
        let options = schema.questions()[0].rendered_options();
        assert_eq!(options.len(), levels, "{minimum}..={maximum}");
        let top = vec![0.0; levels - 1].into_iter().chain([1.0]).collect::<Vec<_>>();
        let answer = Answer::decode(&schema.questions()[0], &top, &config()?)?;
        assert_eq!(schema.project(&[answer])?["field"], json!(maximum));
    }
    Ok(())
}

#[test]
fn keeps_restricted_boolean_constants_and_enums() -> Result<()> {
    for (property, allowed) in [
        (json!({"const": true}), true),
        (json!({"const": false}), false),
        (json!({"enum": [true]}), true),
        (json!({"type": "boolean", "enum": [false]}), false),
    ] {
        let schema = DecisionSchema::from_json_schema(
            &json!({"type": "object", "properties": {"field": property}}),
        )?;
        let question = &schema.questions()[0];
        assert_eq!(question.rendered_options().len(), 1, "{property}");
        let answer = Answer::decode(question, &[-5.0], &config()?)?;
        assert_eq!(schema.project(&[answer])?["field"], json!(allowed), "{property}");
    }
    let free = DecisionSchema::from_json_schema(
        &json!({"type": "object", "properties": {"field": {"enum": [false, true]}}}),
    )?;
    assert_eq!(free.questions()[0].instruction(), "Is `field` true?");
    Ok(())
}

#[test]
fn rejects_validation_keywords_outside_the_supported_subset() {
    for property in [
        json!({"type": "integer", "minimum": 0, "maximum": 4, "multipleOf": 2}),
        json!({"type": "integer", "minimum": 0, "maximum": 4, "exclusiveMaximum": 4}),
        json!({"type": "string", "enum": ["a", "b"], "pattern": "^a$"}),
        json!({"$ref": "#/$defs/Choice"}),
        json!({"const": "a", "enum": ["a", "b"]}),
        json!({"type": "string", "enum": ["a", 1]}),
        json!({"type": "boolean", "const": "yes"}),
        json!({"enum": [1, 2], "minimum": 2}),
        json!({"type": "boolean", "maximum": 1}),
        json!({"anyOf": [{"type": "boolean"}, {"type": "null"}], "type": "boolean"}),
        json!({"anyOf": [{"type": "integer", "minimum": 0, "maximum": 3, "multipleOf": 3}]}),
        json!({"oneOf": [{"enum": [null, true]}, {"type": "null"}]}),
        json!({"oneOf": [{"type": "null"}, {"const": null}]}),
        json!({"oneOf": [{"anyOf": [{"enum": ["a", null]}]}, {"type": "null"}]}),
    ] {
        let schema = json!({"type": "object", "properties": {"field": property}});
        assert!(DecisionSchema::from_json_schema(&schema).is_err(), "{property}");
    }
    for top in [
        json!({"type": "object", "properties": {"a": {"type": "boolean"}}, "minProperties": 1}),
        json!({"type": "object", "properties": {"a": {"type": "boolean"}}, "required": ["b"]}),
    ] {
        assert!(DecisionSchema::from_json_schema(&top).is_err(), "{top}");
    }
}

#[test]
fn accepts_annotations_and_matching_types() -> Result<()> {
    let schema = DecisionSchema::from_json_schema(&json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Decision",
        "type": "object",
        "required": ["kind", "level"],
        "additionalProperties": false,
        "properties": {
            "kind": {"type": ["string", "null"], "enum": ["a", null], "title": "Kind"},
            "level": {"type": "integer", "minimum": 0, "maximum": 2, "default": 1},
            "maybe": {"anyOf": [{"type": "boolean"}, {"type": "null"}], "default": null},
            "one": {"oneOf": [{"enum": ["a", "b"]}, {"type": "null"}]},
            "any": {"anyOf": [{"enum": [null, true]}, {"type": "null"}]},
            "sole": {"oneOf": [{"enum": [null, true]}]}
        }
    }))?;
    assert_eq!(schema.fields().len(), 6);
    Ok(())
}

#[test]
fn ignores_annotations_without_changing_the_plan() -> Result<()> {
    let plain = DecisionSchema::from_json_schema(&json!({
        "type": "object",
        "properties": {
            "kind": {"enum": ["a", "b"]},
            "level": {"type": "integer", "minimum": 0, "maximum": 2}
        }
    }))?;
    let annotated = DecisionSchema::from_json_schema(&json!({
        "$id": "https://example.com/decision",
        "type": "object",
        "properties": {
            "kind": {"enum": ["a", "b"], "$id": "#kind", "format": "token", "readOnly": true},
            "level": {
                "type": "integer", "minimum": 0, "maximum": 2,
                "writeOnly": false, "nullable": true, "format": "int32"
            }
        }
    }))?;
    assert_eq!(annotated, plain);
    Ok(())
}
