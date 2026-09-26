use llguidance::{Matcher, ParserFactory, api::TopLevelGrammar};
use serde_json::{Value, json};

use super::grammar;
mod reasoning;
use crate::model::constraints::tool_tokenizer;

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn factory(tool_end: bool) -> TestResult<ParserFactory> {
    let mut vocab = serde_json::Map::new();
    let mut next = 256_u32;
    for byte in 0..256_u32 {
        let visible = (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174;
        let point = if visible {
            byte
        } else {
            let value = next;
            next += 1;
            value
        };
        let character = char::from_u32(point).ok_or("invalid test alphabet")?;
        vocab.insert(character.to_string(), json!(byte));
    }
    vocab.insert("<|endoftext|>".into(), json!(256));
    let mut serialized = json!({
        "version":"1.0", "truncation":null,"padding":null,"normalizer":null,"post_processor":null,
        "added_tokens":[{"id":256,"content":"<|endoftext|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "model":{"type":"BPE","vocab":vocab,"merges":[]}
    });
    if tool_end {
        serialized["model"]["vocab"]["</tool_call>"] = json!(257);
        serialized["added_tokens"]
            .as_array_mut()
            .ok_or("missing added tokens")?
            .push(json!({
                "id":257,"content":"</tool_call>","single_word":false,"lstrip":false,
                "rstrip":false,"normalized":false,"special":false
            }));
    }
    let env = tool_tokenizer(
        &serialized.to_string(),
        if tool_end {
            258
        } else {
            257
        },
        &[256],
    )?;
    Ok(ParserFactory::new_simple(&env)?)
}

fn accepts(schema: &Value, continuation: &str) -> TestResult<bool> {
    let factory = factory(false)?;
    let mut matcher = Matcher::new(
        factory.create_parser(TopLevelGrammar::from_lark(grammar::compile(schema, None)?)),
    );
    for token in factory.tok_env().tokenize(continuation) {
        if !matcher.compute_mask()?.is_allowed(token) {
            return Ok(false);
        }
        matcher.consume_token(token)?;
    }
    Ok(matcher.is_accepting()?)
}

#[test]
fn masks_invalid_json_enums_nulls_and_cardinality() -> TestResult {
    let schema = json!({"type":"object","required":["items"],"additionalProperties":false,"properties":{
        "items":{"type":"array","minItems":1,"maxItems":1,"items":{"type":"object","additionalProperties":false,
            "required":["kind","unknown"],"properties":{"kind":{"enum":["launch"]},"unknown":{"type":"null"}}}}
    }});
    let valid = "<parameter=items>[{\"kind\":\"launch\",\"unknown\":null}]</parameter>\n</function>\n</tool_call>";
    assert!(accepts(&schema, valid)?);
    for bad in [
        valid.replace(",\"unknown\"", "\"unknown\""),
        valid.replace("launch", "finance"),
        valid.replace("null", "123"),
        valid.replace("}]", "},{}]"),
        valid.replace(",\"unknown\":null", ""),
        valid.replace("</tool_call>", ""),
    ] {
        assert!(!accepts(&schema, &bad)?, "accepted invalid schema output");
    }
    Ok(())
}

#[test]
fn strings_support_unicode_escapes_and_optional_parameters() -> TestResult {
    let schema =
        json!({"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"]});
    let text =
        "<parameter=summary>\"Zażółć 🥔 \\\"quoted\\\"\"</parameter>\n</function>\n</tool_call>";
    assert!(accepts(&schema, text)?);
    assert!(!accepts(&schema, &text.replace("\\\"quoted\\\"", "\"quoted\""))?);
    let optional = json!({"type":"object","properties":{"summary":{"type":"string"}}});
    assert!(accepts(&optional, "</function>\n</tool_call>")?);
    assert!(grammar::compile(&json!({"type":"object","properties":{},"allOf":[]}), None).is_err());
    Ok(())
}

#[test]
fn named_tool_closes_with_added_token_in_real_tokenizer_shape() -> TestResult {
    let factory = factory(true)?;
    let end = factory.tok_env().tok_trie().get_special_token("</tool_call>");
    assert_eq!(end, None);
    let schema = json!({"type":"object","properties":{"values":{"type":"array","items":{"type":"object"}}},"required":["values"]});
    let grammar = grammar::compile(&schema, end)?;
    let mut matcher = Matcher::new(factory.create_parser(TopLevelGrammar::from_lark(grammar)));
    let text = "<parameter=values>[{\"value\":1,\"other\":2}]</parameter>\n</function>\n";
    for token in factory.tok_env().tokenize(text).into_iter().chain([257]) {
        assert!(matcher.compute_mask()?.is_allowed(token));
        matcher.consume_token(token)?;
    }
    assert!(matcher.is_accepting()?);
    Ok(())
}

#[test]
fn ordinary_added_tokens_remain_text_inside_json_values() -> TestResult {
    let factory = factory(true)?;
    let schema = json!({"type":"object","properties":{"text":{"const":"Literal </tool_call>"}},"required":["text"]});
    let grammar = grammar::compile(&schema, None)?;
    let mut matcher = Matcher::new(factory.create_parser(TopLevelGrammar::from_lark(grammar)));
    let text = "<parameter=text>\"Literal </tool_call>\"</parameter>\n</function>\n</tool_call>";
    let tokens = factory.tok_env().tokenize(text);
    assert_eq!(tokens.iter().filter(|token| **token == 257).count(), 2);
    for token in tokens {
        assert!(matcher.compute_mask()?.is_allowed(token));
        matcher.consume_token(token)?;
    }
    assert!(matcher.is_accepting()?);
    Ok(())
}

#[test]
fn envelope_keeps_native_whitespace_before_and_after_json() -> TestResult {
    let schema = json!({"type":"object","properties":{"events":{"type":"array","items":{"type":"object","properties":{"kind":{"enum":["offer"]}},"required":["kind"],"additionalProperties":false}}},"required":["events"]});
    let text =
        "<parameter=events>\n[{\"kind\": \"offer\"}]\n</parameter>\n</function>\n</tool_call>";
    assert!(accepts(&schema, text)?);
    assert!(accepts(&schema, &text.replace('\n', ""))?);
    assert!(!accepts(&schema, &text.replace("offer", "invented"))?);
    Ok(())
}

#[test]
fn formatting_gaps_are_bounded_without_truncating_string_data() -> TestResult {
    let schema = json!({"type":"object","properties":{"review":{"type":"object",
        "required":["explanation"],"properties":{"explanation":{"type":"string"}}}},"required":["review"]});
    let spaces = " ".repeat(256);
    let valid = format!(
        "<parameter=review>\n{{\"explanation\": \"{spaces}\"}}\n</parameter>\n</function>\n</tool_call>"
    );
    assert!(accepts(&schema, &valid)?);
    for bad in [
        valid.replacen(">\n{", &format!(">{spaces}{{"), 1),
        valid.replacen("{\"explanation", &format!("{{{spaces}\"explanation"), 1),
        valid.replacen(": ", &format!(":{spaces}"), 1),
        valid.replacen("}\n</parameter>", &format!("}}{spaces}</parameter>"), 1),
        valid.replace("</function>\n", &format!("</function>{spaces}")),
    ] {
        assert!(!accepts(&schema, &bad)?, "accepted an unbounded formatting gap");
    }
    Ok(())
}
