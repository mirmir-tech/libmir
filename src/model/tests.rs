use foundation::protocol::ChatMessage;
use models::chat::TemplateKind;

use super::*;

#[test]
fn derives_a_stable_model_identifier() -> Result<()> {
    assert_eq!(model_id(Path::new("/models/example"))?, "example");
    Ok(())
}

#[test]
fn rejects_a_request_larger_than_the_model_context() {
    assert!(matches!(
        validate_context(900, 200, 1_024),
        Err(Error::Context { requested: 1_100, .. })
    ));
}

#[test]
#[ignore = "loads a real Gemma 4 checkpoint; set MIRMIR_GEMMA4_MODEL"]
fn prepares_real_gemma4_turn_protocol_like_mlx_lm() -> Result<()> {
    let path = std::env::var_os("MIRMIR_GEMMA4_MODEL")
        .ok_or(Error::MissingEnvironment("MIRMIR_GEMMA4_MODEL"))?;
    let descriptor = ModelDescriptor::inspect(path, GenerationOverrides::default())?;
    let request = ChatCompletionRequest {
        model: "gemma4".into(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Napisz jedno krótkie zdanie po polsku.".into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: None,
        stream: false,
        max_tokens: Some(32),
        temperature: Some(0.0),
        top_p: None,
        top_k: None,
        repetition_penalty: None,
        seed: None,
    };

    let prepared = descriptor.prepare(&request)?;

    assert_eq!(descriptor.template().kind(), TemplateKind::TurnDelimited);
    assert_eq!(
        prepared.prompt.text,
        "<|turn>user\nNapisz jedno krótkie zdanie po polsku.<turn|>\n<|turn>assistant\n"
    );
    assert_eq!(
        prepared.tokens.token_ids,
        [
            105, 2364, 107, 236_797, 20784, 236_802, 81155, 217_074, 703, 19944, 10255, 2099, 1268,
            57217, 236_761, 106, 107, 105, 111_457, 107,
        ]
    );
    Ok(())
}

#[test]
#[ignore = "loads a real Qwen 3.5/3.6 checkpoint; set MIRMIR_QWEN35_MODEL"]
fn prepares_real_qwen35_chatml_fallback() -> Result<()> {
    let path = std::env::var_os("MIRMIR_QWEN35_MODEL")
        .ok_or(Error::MissingEnvironment("MIRMIR_QWEN35_MODEL"))?;
    let descriptor = ModelDescriptor::inspect(path, GenerationOverrides::default())?;
    let request = ChatCompletionRequest {
        model: "qwen35".into(),
        messages: vec![ChatMessage {
            role: "user".into(),
            content: "Napisz jedno zdanie o Warszawie.".into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: None,
        stream: false,
        max_tokens: Some(32),
        temperature: Some(0.0),
        top_p: None,
        top_k: None,
        repetition_penalty: None,
        seed: None,
    };

    let prepared = descriptor.prepare(&request)?;

    assert_eq!(descriptor.template().kind(), TemplateKind::ChatMl);
    assert_eq!(
        prepared.prompt.text,
        "<|im_start|>user\nNapisz jedno zdanie o Warszawie.<|im_end|>\n<|im_start|>assistant\n"
    );
    assert_eq!(
        prepared.tokens.token_ids,
        [
            248_045, 846, 198, 45, 13_331, 89, 180_428, 35_051, 18_571, 296, 224_500, 13, 248_046,
            198, 248_045, 74_455, 198,
        ]
    );
    Ok(())
}
