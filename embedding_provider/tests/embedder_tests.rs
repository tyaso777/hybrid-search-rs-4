use embedding_provider::config::{default_stdio_config, ONNX_STDIO_DEFAULTS};
use embedding_provider::embedder::{
    Embedder, EmbedderError, OnnxHttpConfig, OnnxHttpEmbedder, OnnxStdIoConfig, OnnxStdIoEmbedder,
    ProviderKind,
};

fn stdio_config(max_input_length: usize) -> OnnxStdIoConfig {
    let mut config = default_stdio_config();
    config.max_input_length = max_input_length;
    config
}

fn assert_vectors_close(lhs: &[f32], rhs: &[f32]) {
    assert_eq!(lhs.len(), rhs.len(), "vector lengths differ");
    for (index, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff <= 1e-4,
            "vectors diverge at position {index}: {a} vs {b} (diff {diff})"
        );
    }
}

#[test]
fn stdio_embedder_produces_deterministic_vectors() {
    let embedder = OnnxStdIoEmbedder::new(stdio_config(ONNX_STDIO_DEFAULTS.max_input_tokens))
        .expect("configuration is valid and model loads");

    let sentence = "Rust makes systems programming safer without sacrificing speed.";
    let vector_a = embedder.embed(sentence).expect("first embedding succeeds");
    let vector_b = embedder.embed(sentence).expect("second embedding succeeds");

    assert_eq!(vector_a.len(), ONNX_STDIO_DEFAULTS.embedding_dimension);
    assert_vectors_close(&vector_a, &vector_b);
    assert!(
        vector_a.iter().any(|component| component.abs() > 1e-3),
        "embedding should not be all zeros"
    );

    let info = embedder.info();
    assert_eq!(info.provider, ProviderKind::OnnxStdIo);
    assert_eq!(info.dimension, ONNX_STDIO_DEFAULTS.embedding_dimension);
    assert_eq!(
        info.embedding_model_id,
        ONNX_STDIO_DEFAULTS.embedding_model_id
    );
}

#[test]
fn embed_batch_matches_individual_embeddings() {
    let embedder = OnnxStdIoEmbedder::new(stdio_config(ONNX_STDIO_DEFAULTS.max_input_tokens))
        .expect("configuration is valid and model loads");

    let inputs = [
        "embeddings unlock semantic search",
        "hybrid ranking mixes bm25 and vectors",
    ];
    let batch_vectors = embedder
        .embed_batch(&inputs)
        .expect("batch embedding succeeds");

    assert_eq!(batch_vectors.len(), inputs.len());

    for (input, batch_vector) in inputs.iter().zip(batch_vectors.iter()) {
        let single = embedder.embed(input).expect("single embedding succeeds");
        assert_vectors_close(&single, batch_vector);
    }
}

#[test]
fn enforcing_max_input_length_returns_error() {
    let embedder =
        OnnxStdIoEmbedder::new(stdio_config(8)).expect("configuration is valid and model loads");
    let too_long = "rust ".repeat(64);

    let err = embedder
        .embed(&too_long)
        .expect_err("inputs exceeding max tokens should fail");

    match err {
        EmbedderError::InputTooLong {
            max_length,
            actual_length,
        } => {
            assert_eq!(max_length, 8);
            assert!(actual_length > max_length);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn http_embedder_reports_provider_metadata_and_handles_empty_batch() {
    let config = OnnxHttpConfig {
        endpoint: "http://localhost:9000/embed".into(),
        auth_token: Some("token-123".into()),
        dimension: 12,
        max_input_length: 1024,
        embedding_model_id: "mock-onnx-http".into(),
        text_repr_version: "v2".into(),
    };
    let embedder = OnnxHttpEmbedder::new(config).expect("configuration is valid");

    let info = embedder.info();
    assert_eq!(info.provider, ProviderKind::OnnxHttp);
    assert_eq!(info.dimension, 12);
    assert_eq!(info.embedding_model_id, "mock-onnx-http");

    let empty: [&str; 0] = [];
    let batch = embedder
        .embed_batch(&empty)
        .expect("empty batches should be allowed");
    assert!(batch.is_empty());
}

