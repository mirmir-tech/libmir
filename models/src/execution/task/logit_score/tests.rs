#![cfg(test)]

use serde_json::json;

use super::*;

#[test]
fn discovers_scoring_config_and_instruction() -> Result<()> {
    let modules = json!([{"path":"1_LogitScore","type":"sentence_transformers.cross_encoder.modules.logit_score.LogitScore"}]);
    assert_eq!(
        CausalScoringTask::config_path(&modules)?.as_deref(),
        Some("1_LogitScore/config.json")
    );
    let task = CausalScoringTask::from_values(
        &json!({"true_token_id":9693,"false_token_id":2152}),
        Some(&json!({"default_prompt_name":"query","prompts":{"query":"Find relevant passages"}})),
        151_936,
    )?;
    assert_eq!(task.instruction.as_deref(), Some("Find relevant passages"));
    Ok(())
}

#[test]
fn rejects_ambiguous_or_invalid_contracts() {
    for path in ["", "../escape", "/absolute"] {
        assert!(
            CausalScoringTask::config_path(&json!([{"path":path,"type":"a.LogitScore"}])).is_err()
        );
    }
    for (yes, no) in [(1, 1), (10, 2), (1, 10)] {
        assert!(
            CausalScoringTask::from_values(
                &json!({"true_token_id":yes,"false_token_id":no}),
                None,
                10
            )
            .is_err()
        );
    }
    assert!(
        CausalScoringTask::from_values(
            &json!({"true_token_id":1,"false_token_id":2}),
            Some(&json!({"default_prompt_name":"missing"})),
            10
        )
        .is_err()
    );
}
