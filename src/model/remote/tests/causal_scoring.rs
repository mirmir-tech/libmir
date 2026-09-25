use super::*;

#[test]
fn causal_scoring_retains_decoder_admission_and_task_identity() -> Result<()> {
    let modules = json!([{"path":"1_LogitScore","type":"cross_encoder.LogitScore"}]);
    let score = json!({"true_token_id":2,"false_token_id":1});
    let metadata = RemoteTaskMetadata {
        modules: Some(&modules),
        logit_score: Some(&score),
        ..Default::default()
    };
    let contract = RemoteModelContract::inspect(&decoder_config(), &dense_catalog(), metadata)?
        .ok_or_else(|| {
            Error::Model(models::ModelsError::InvalidConfig("scoring contract missing".into()))
        })?;
    assert!(matches!(contract.task(), TaskExecutionPlan::CausalScoring { .. }));
    assert!(
        contract
            .architecture_requirements()
            .capabilities
            .contains(&models::execution::ArchitectureCapability::CausalScoringTask)
    );
    for backend in [BackendTarget::Metal, BackendTarget::Cuda] {
        assert_eq!(contract.admission(backend).status, AdmissionStatus::Supported);
    }
    let missing = RemoteTaskMetadata { logit_score: None, ..metadata };
    assert!(RemoteModelContract::inspect(&decoder_config(), &dense_catalog(), missing).is_err());
    Ok(())
}
