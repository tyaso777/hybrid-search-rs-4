use std::cmp::min;

use embedding_provider::config::{default_stdio_config, ONNX_STDIO_DEFAULTS};
use embedding_provider::embedder::{Embedder, OnnxStdIoEmbedder};

fn main() {
    let text = std::env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_owned();

    let input = if text.is_empty() {
        "sample text for embedding".to_string()
    } else {
        text
    };

    let config = default_stdio_config();
    println!("model path: {}", ONNX_STDIO_DEFAULTS.model_path);
    println!("runtime dll: {}", ONNX_STDIO_DEFAULTS.runtime_dll_path);

    let embedder = OnnxStdIoEmbedder::new(config).expect("failed to initialize embedder");
    let vector = embedder.embed(&input).expect("embedding failed");

    println!("input: {input}");
    println!("vector length: {}", vector.len());

    let preview = &vector[..min(8, vector.len())];
    println!("first {} values: {preview:?}", preview.len());
}

