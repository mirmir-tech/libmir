use foundation::conversation::Message;
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
    let request = Conversation {
        messages: vec![Message {
            role: "user".into(),
            content: "Napisz jedno krótkie zdanie po polsku.".into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: foundation::conversation::ToolChoice::default(),
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
#[ignore = "loads a real Gemma 4 checkpoint; set MIRMIR_GEMMA4_MODEL"]
fn prepares_real_gemma4_message_boundary_checkpoint() -> Result<()> {
    let path = std::env::var_os("MIRMIR_GEMMA4_MODEL")
        .ok_or(Error::MissingEnvironment("MIRMIR_GEMMA4_MODEL"))?;
    let descriptor = ModelDescriptor::inspect(path, GenerationOverrides::default())?;
    let mut request = Conversation {
        messages: Vec::new(),
        tools: Vec::new(),
        tool_choice: foundation::conversation::ToolChoice::default(),
    };
    for (role, content) in [("system", "Stały kontekst."), ("user", "Odpowiedz krótko.")] {
        request.messages.push(Message {
            role: role.into(),
            content: content.into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        });
    }

    let prepared = descriptor.prepare(&request)?;

    assert_eq!(prepared.cache_checkpoints.len(), 1);
    assert!(prepared.cache_checkpoints[0] < prepared.tokens.token_ids.len());
    Ok(())
}

#[test]
#[ignore = "loads a real Qwen 3.5/3.6 checkpoint; set MIRMIR_QWEN35_MODEL"]
fn prepares_real_qwen35_multi_turn_checkpoint() -> Result<()> {
    let path = std::env::var_os("MIRMIR_QWEN35_MODEL")
        .ok_or(Error::MissingEnvironment("MIRMIR_QWEN35_MODEL"))?;
    let descriptor = ModelDescriptor::inspect(path, GenerationOverrides::default())?;
    let mut request = Conversation {
        messages: vec![Message {
            role: "user".into(),
            content: "Napisz jedno zdanie o Warszawie.".into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }],
        tools: Vec::new(),
        tool_choice: foundation::conversation::ToolChoice::default(),
    };

    let prepared = descriptor.prepare(&request)?;
    assert_ne!(prepared.tokens.token_ids, Vec::<u32>::new());
    assert_eq!(prepared.cache_checkpoints, Vec::<usize>::new());

    request.messages.extend([
        Message {
            role: "assistant".into(),
            content: "Warszawa jest stolicą Polski.".into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        },
        Message {
            role: "user".into(),
            content: "Rozwiń odpowiedź.".into(),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        },
    ]);
    let multi_turn = descriptor.prepare(&request)?;
    assert_eq!(multi_turn.cache_checkpoints.len(), 1);
    assert!(multi_turn.cache_checkpoints[0] < multi_turn.tokens.token_ids.len());
    Ok(())
}
