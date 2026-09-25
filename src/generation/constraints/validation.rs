use foundation::conversation::FunctionCall;
use serde_json::Value;

use crate::{Error, Result};

/// Uniqueness depends on prior values, so enforce it on the complete arguments.
/// All other decoder constraints remain intact; the original schema is always
/// validated before a completed tool call is returned to the application.
pub(super) fn decoding_schema(mut schema: Value) -> Value {
    if let Some(fields) = schema.as_object_mut() {
        fields.remove("uniqueItems");
        for keyword in ["properties", "patternProperties", "$defs", "definitions"] {
            if let Some(children) = fields.get_mut(keyword).and_then(Value::as_object_mut) {
                for child in children.values_mut() {
                    *child = decoding_schema(child.take());
                }
            }
        }
        for keyword in [
            "items",
            "additionalItems",
            "additionalProperties",
            "unevaluatedItems",
            "unevaluatedProperties",
            "contains",
            "propertyNames",
            "not",
            "if",
            "then",
            "else",
            "allOf",
            "anyOf",
            "oneOf",
            "prefixItems",
        ] {
            if let Some(child) = fields.get_mut(keyword) {
                if let Some(children) = child.as_array_mut() {
                    for child in children {
                        *child = decoding_schema(child.take());
                    }
                } else {
                    *child = decoding_schema(child.take());
                }
            }
        }
    }
    schema
}

pub(super) fn validate(validator: &jsonschema::Validator, output: &str) -> Result<()> {
    #[derive(serde::Deserialize)]
    struct Call {
        function: FunctionCall,
    }
    let calls: Vec<Call> =
        serde_json::from_str(output).map_err(|error| Error::InvalidToolCall(error.to_string()))?;
    for call in calls {
        validator.validate(&call.function.arguments).map_err(|error| {
            Error::InvalidToolCall(format!(
                "tool arguments violate schema at {} (constraint {})",
                error.instance_path, error.schema_path
            ))
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn unique_items_are_enforced_after_generation_without_changing_constants() -> Result<()> {
        let schema = json!({"type":"object","properties":{
            "ids":{"type":"array","uniqueItems":true,"items":{"type":"string"}},
            "literal":{"const":{"uniqueItems":true}},
            "uniqueItems":{"type":"boolean"}
        },"required":["ids"]});
        let decoding = decoding_schema(schema.clone());
        assert!(decoding["properties"]["ids"].get("uniqueItems").is_none());
        assert_eq!(decoding["properties"]["literal"], schema["properties"]["literal"]);
        assert_eq!(decoding["properties"]["uniqueItems"], json!({"type":"boolean"}));
        let validator = jsonschema::validator_for(&schema)
            .map_err(|error| Error::InvalidToolCall(error.to_string()))?;
        let calls = |ids| {
            json!([{"id":"one","type":"function","function":{"name":"report","arguments":{"ids":ids}}}]).to_string()
        };
        assert!(validate(&validator, &calls(json!(["a", "b"]))).is_ok());
        let error = validate(&validator, &calls(json!(["a", "a"])))
            .err()
            .ok_or_else(|| Error::InvalidToolCall("duplicate accepted".into()))?;
        assert!(error.to_string().contains("uniqueItems"));
        assert!(validate(&validator, &calls(json!([1]))).is_err());
        Ok(())
    }
}
