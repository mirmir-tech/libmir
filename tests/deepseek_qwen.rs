use libmir::{
    Conversation, Error, GenerationOverrides, GenerationRequest, Library, Message, Result,
    RuntimeConfig,
};

#[test]
#[ignore = "loads DeepSeek-R1-0528-Qwen3-8B; set DEEPSEEK_QWEN_MODEL"]
fn preserves_deepseek_qwen_greedy_digest() -> Result<()> {
    let path = std::env::var_os("DEEPSEEK_QWEN_MODEL")
        .ok_or(Error::MissingEnvironment("DEEPSEEK_QWEN_MODEL"))?;
    let model = Library::new(RuntimeConfig::default()).load(
        path,
        GenerationOverrides::default(),
        &mut |_event| {},
    )?;
    let output = model.generate(
        &GenerationRequest {
            conversation: Conversation {
                messages: vec![Message {
                    role: "user".into(),
                    content: "Hello".into(),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                }],
                tools: Vec::new(),
                tool_choice: libmir::ToolChoice::default(),
            },
            options: GenerationOverrides {
                max_tokens: Some(16),
                temperature: Some(0.0),
                top_p: Some(1.0),
                top_k: Some(0),
                repetition_penalty: Some(1.0),
                ..GenerationOverrides::default()
            },
            seed: None,
            reasoning_cycle: libmir::ReasoningCyclePolicy::default(),
        },
        &mut |_event| {},
        &mut |_token| {},
    )?;

    assert_eq!(
        output.token_ids,
        [
            151_667, 198, 32_313, 11, 279, 1_196, 1_101, 1_053, 1_036, 9_707, 854, 448, 264, 4_285,
            42_113, 13
        ]
    );
    Ok(())
}
