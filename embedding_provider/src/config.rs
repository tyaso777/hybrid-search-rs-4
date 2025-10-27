use std::path::PathBuf;

use crate::embedder::OnnxStdIoConfig;

/// Default settings for the local ONNX embedder.
#[derive(Debug, Clone, Copy)]
pub struct OnnxStdIoDefaults {
    pub model_path: &'static str,
    pub tokenizer_path: &'static str,
    pub runtime_dll_path: &'static str,
    pub embedding_dimension: usize,
    pub max_input_tokens: usize,
    pub embedding_model_id: &'static str,
    pub text_repr_version: &'static str,
}

/// Shared defaults so CLI・GUI・tests can stay in sync.
pub const ONNX_STDIO_DEFAULTS: OnnxStdIoDefaults = OnnxStdIoDefaults {
    model_path: "models/ruri-v3-onnx/model.onnx",
    tokenizer_path: "models/ruri-v3-onnx/tokenizer.json",
    runtime_dll_path: "bin/onnxruntime-win-x64-1.23.1/lib/onnxruntime.dll",
    embedding_dimension: 768,
    max_input_tokens: 8192,
    embedding_model_id: "ruri-v3-onnx",
    text_repr_version: "v1",
};

/// Convenience helper to build an [`OnnxStdIoConfig`] from the shared defaults.
pub fn default_stdio_config() -> OnnxStdIoConfig {
    // Resolve asset paths relative to this crate's directory, so it works
    // regardless of the current working directory (workspace root or crate dir).
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    OnnxStdIoConfig {
        model_path: base.join(ONNX_STDIO_DEFAULTS.model_path),
        tokenizer_path: base.join(ONNX_STDIO_DEFAULTS.tokenizer_path),
        runtime_library_path: base.join(ONNX_STDIO_DEFAULTS.runtime_dll_path),
        dimension: ONNX_STDIO_DEFAULTS.embedding_dimension,
        max_input_length: ONNX_STDIO_DEFAULTS.max_input_tokens,
        embedding_model_id: ONNX_STDIO_DEFAULTS.embedding_model_id.into(),
        text_repr_version: ONNX_STDIO_DEFAULTS.text_repr_version.into(),
        preload_model_to_memory: false,
    }
}
