use libmir::{
    EmbeddingRequest, GenerationOverrides, Library, RerankRequest, Result, RuntimeConfig,
};

#[test]
#[ignore = "loads a real checkpoint; set EMBEDDING_MODEL"]
fn embeds_through_the_public_library_api() -> Result<()> {
    let path = std::env::var_os("EMBEDDING_MODEL")
        .ok_or(libmir::Error::MissingEnvironment("EMBEDDING_MODEL"))?;
    let model = Library::new(RuntimeConfig::default()).load(
        path,
        GenerationOverrides::default(),
        &mut |_event| {},
    )?;
    let output = model.embed(EmbeddingRequest {
        inputs: vec!["weather forecast".into(), "sunny afternoon".into()],
        dimensions: Some(128),
        prompt_name: Some("query".into()),
    })?;

    assert_eq!(output.embeddings.len(), 2);
    assert!(output.embeddings.iter().all(|embedding| embedding.len() == 128));
    assert!(output.embeddings.iter().all(|embedding| {
        let norm = embedding.iter().map(|value| value * value).sum::<f32>().sqrt();
        (norm - 1.0).abs() < 0.02
    }));
    assert!(output.prompt_tokens > 0);
    Ok(())
}

#[test]
#[ignore = "loads a real checkpoint; set RERANK_MODEL"]
fn reranks_through_the_public_library_api() -> Result<()> {
    let path = std::env::var_os("RERANK_MODEL")
        .ok_or(libmir::Error::MissingEnvironment("RERANK_MODEL"))?;
    let model = Library::new(RuntimeConfig::default()).load(
        path,
        GenerationOverrides::default(),
        &mut |_event| {},
    )?;
    let output = model.rerank(RerankRequest {
        query: "weather forecast".into(),
        documents: vec!["Tomorrow will be sunny.".into(), "A history of typography.".into()],
        max_length: None,
        raw_scores: false,
    })?;

    assert_eq!(output.results.len(), 2);
    assert!(output.results.windows(2).all(|pair| pair[0].score >= pair[1].score));
    assert!(output.results.iter().all(|result| (0.0..=1.0).contains(&result.score)));
    assert!(output.prompt_tokens > 0);
    Ok(())
}
