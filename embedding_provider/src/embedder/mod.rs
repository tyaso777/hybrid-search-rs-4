use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use ndarray::Array2;
use ort::{Error as OrtError, session::Session, value::Tensor};
use thiserror::Error;
use tokenizers::{Encoding, Tokenizer};

/// Identifies the backing implementation that powers an embedder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderKind {
    OnnxStdIo,
    OnnxHttp,
}

/// Static metadata describing a particular embedder instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbedderInfo {
    pub provider: ProviderKind,
    pub embedding_model_id: String,
    pub dimension: usize,
    pub text_repr_version: String,
}

/// Errors that can be produced by embedder operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EmbedderError {
    #[error("invalid embedder configuration: {message}")]
    InvalidConfiguration { message: String },
    #[error("input text exceeds max length of {max_length} tokens, actual length: {actual_length}")]
    InputTooLong {
        max_length: usize,
        actual_length: usize,
    },
    #[error("provider failure: {message}")]
    ProviderFailure { message: String },
}

/// Core interface for all embedder implementations.
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError>;
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError>;
    fn info(&self) -> &EmbedderInfo;
}

/// Configuration for a local ONNX embedder driven through stdio.
#[derive(Debug, Clone)]
pub struct OnnxStdIoConfig {
    pub model_path: PathBuf,
    pub runtime_library_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub dimension: usize,
    pub max_input_length: usize,
    pub embedding_model_id: String,
    pub text_repr_version: String,
}

/// ONNX-based embedder that executes models through the ONNX Runtime shared library.
#[derive(Debug)]
pub struct OnnxStdIoEmbedder {
    info: EmbedderInfo,
    session: Mutex<Session>,
    tokenizer: Arc<Tokenizer>,
    pad_id: i64,
    max_input_length: usize,
}

#[derive(Debug)]
struct PreparedBatch {
    input_ids: Tensor<i64>,
    attention_mask: Tensor<i64>,
    attention_rows: Vec<Vec<i64>>,
}

static ORT_RUNTIME_PATH: OnceLock<PathBuf> = OnceLock::new();

impl OnnxStdIoEmbedder {
    pub fn new(config: OnnxStdIoConfig) -> Result<Self, EmbedderError> {
        if config.dimension == 0 {
            return Err(EmbedderError::InvalidConfiguration {
                message: "dimension must be greater than zero".into(),
            });
        }

        if config.max_input_length == 0 {
            return Err(EmbedderError::InvalidConfiguration {
                message: "max_input_length must be greater than zero".into(),
            });
        }

        let runtime_library_path =
            resolve_existing_path(&config.runtime_library_path, "ONNX Runtime shared library")?;

        ensure_ort_initialized(&runtime_library_path)?;

        let model_path = resolve_existing_path(&config.model_path, "ONNX model")?;
        let tokenizer_path = resolve_existing_path(&config.tokenizer_path, "tokenizer config")?;

        let session = Session::builder()
            .map_err(|err| map_ort_error("create session builder", err))?
            .commit_from_file(&model_path)
            .map_err(|err| map_ort_error("load ONNX model", err))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| map_tokenizer_error("load tokenizer", err))?;

        let pad_id =
            tokenizer
                .token_to_id("<pad>")
                .ok_or_else(|| EmbedderError::InvalidConfiguration {
                    message: format!(
                        "tokenizer `{}` does not declare a `<pad>` token",
                        tokenizer_path.display()
                    ),
                })? as i64;

        let info = EmbedderInfo {
            provider: ProviderKind::OnnxStdIo,
            embedding_model_id: config.embedding_model_id,
            dimension: config.dimension,
            text_repr_version: config.text_repr_version,
        };

        Ok(Self {
            info,
            session: Mutex::new(session),
            tokenizer: Arc::new(tokenizer),
            pad_id,
            max_input_length: config.max_input_length,
        })
    }

    fn prepare_encodings(&self, texts: &[&str]) -> Result<Vec<Encoding>, EmbedderError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let tokenizer = &self.tokenizer;
        let encodings = texts
            .iter()
            .map(|t| tokenizer.encode(*t, true))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|err| map_tokenizer_error("tokenize inputs", err))?;

        let max_len = encodings.iter().map(Encoding::len).max().unwrap_or(0);
        if max_len > self.max_input_length {
            return Err(EmbedderError::InputTooLong {
                max_length: self.max_input_length,
                actual_length: max_len,
            });
        }

        Ok(encodings)
    }

    fn build_input_tensors(&self, encodings: &[Encoding]) -> Result<PreparedBatch, EmbedderError> {
        let batch = encodings.len();
        let seq_len = encodings.iter().map(Encoding::len).max().unwrap_or(0);

        let mut input_ids = Array2::<i64>::zeros((batch, seq_len));
        let mut attention_mask = Array2::<i64>::zeros((batch, seq_len));
        let mut attention_rows = Vec::with_capacity(batch);

        for (row, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            for (col, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
                input_ids[(row, col)] = id as i64;
                attention_mask[(row, col)] = m as i64;
            }

            // pad the rest with pad_id and mask 0
            for col in ids.len()..seq_len {
                input_ids[(row, col)] = self.pad_id;
                attention_mask[(row, col)] = 0;
            }

            attention_rows.push(
                (0..seq_len)
                    .map(|i| attention_mask[(row, i)])
                    .collect::<Vec<i64>>(),
            );
        }

        let input_ids = Tensor::from_array(input_ids).map_err(|err| map_ort_error("prepare input_ids", err))?;
        let attention_mask = Tensor::from_array(attention_mask)
            .map_err(|err| map_ort_error("prepare attention_mask", err))?;

        Ok(PreparedBatch {
            input_ids,
            attention_mask,
            attention_rows,
        })
    }

    fn run_session(
        &self,
        input_ids: Tensor<i64>,
        attention_mask: Tensor<i64>,
    ) -> Result<(Vec<f32>, usize, usize, usize), EmbedderError> {
        let mut session = self.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs![input_ids, attention_mask])
            .map_err(|err| map_ort_error("execute ONNX session", err))?;

        // Expect exactly one output tensor (index 0)
        let output = &outputs[0];
        let (shape, data) = output
            .try_extract_tensor::<f32>()
            .map_err(|err| map_ort_error("extract output tensor", err))?;

        if shape.len() != 3 {
            let dims: Vec<i64> = shape.iter().copied().collect();
            return Err(EmbedderError::ProviderFailure {
                message: format!(
                    "model output must be rank-3 [batch, seq_len, hidden], got shape {:?}",
                    dims
                ),
            });
        }

        let batch: usize = shape[0].try_into().unwrap();
        let seq_len: usize = shape[1].try_into().unwrap();
        let hidden: usize = shape[2].try_into().unwrap();

        Ok((data.to_vec(), batch, seq_len, hidden))
    }

    fn mean_pool(
        &self,
        data: &[f32],
        attention_rows: &[Vec<i64>],
        seq_len: usize,
        hidden: usize,
    ) -> Result<Vec<Vec<f32>>, EmbedderError> {
        let batch = attention_rows.len();
        let mut results = Vec::with_capacity(batch);
        for b in 0..batch {
            let mut sum = vec![0f32; hidden];
            let mut count = 0f32;

            for t in 0..seq_len {
                if attention_rows[b][t] == 1 {
                    let base = (b * seq_len + t) * hidden;
                    for h in 0..hidden {
                        sum[h] += data[base + h];
                    }
                    count += 1.0;
                }
            }

            if count > 0.0 {
                for h in 0..hidden {
                    sum[h] /= count;
                }
            }

            results.push(sum);
        }

        Ok(results)
    }
}

impl Embedder for OnnxStdIoEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        let encodings = self.prepare_encodings(&[text])?;
        let prepared = self.build_input_tensors(&encodings)?;

        let (raw_data, batch, seq_len, hidden) =
            self.run_session(prepared.input_ids, prepared.attention_mask)?;

        if batch != 1 {
            return Err(EmbedderError::ProviderFailure {
                message: format!("model returned unexpected batch size {batch}, expected 1 for single input"),
            });
        }

        let pooled = self.mean_pool(&raw_data, &prepared.attention_rows, seq_len, hidden)?;
        let vector = pooled
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::ProviderFailure { message: "missing pooled output".into() })?;

        if vector.len() != self.info.dimension {
            return Err(EmbedderError::ProviderFailure {
                message: format!(
                    "pooled embedding dimension {} does not match configured dimension {}",
                    vector.len(), self.info.dimension
                ),
            });
        }

        Ok(vector)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self.prepare_encodings(texts)?;
        let prepared = self.build_input_tensors(&encodings)?;
        let expected_seq_len = encodings.iter().map(Encoding::len).max().unwrap_or(0);

        let (raw_data, batch, seq_len_from_model, hidden) =
            self.run_session(prepared.input_ids, prepared.attention_mask)?;

        if batch != prepared.attention_rows.len() {
            return Err(EmbedderError::ProviderFailure {
                message: format!(
                    "model returned batch size {batch}, but prepared {} attention masks",
                    prepared.attention_rows.len()
                ),
            });
        }

        if seq_len_from_model != expected_seq_len {
            return Err(EmbedderError::ProviderFailure {
                message: format!(
                    "model returned sequence length {seq_len_from_model}, expected {expected_seq_len}"
                ),
            });
        }

        if prepared
            .attention_rows
            .iter()
            .any(|row| row.len() != expected_seq_len)
        {
            return Err(EmbedderError::ProviderFailure {
                message: "internal padding row length mismatch".into(),
            });
        }

        self.mean_pool(&raw_data, &prepared.attention_rows, expected_seq_len, hidden)
    }

    fn info(&self) -> &EmbedderInfo {
        &self.info
    }
}

/// Configuration for an ONNX embedder exposed through an HTTP endpoint.
#[derive(Debug, Clone)]
pub struct OnnxHttpConfig {
    pub endpoint: String,
    pub auth_token: Option<String>,
    pub dimension: usize,
    pub max_input_length: usize,
    pub embedding_model_id: String,
    pub text_repr_version: String,
}

/// Deterministic pseudo embedder representing an ONNX model behind HTTP.
#[derive(Debug, Clone)]
pub struct OnnxHttpEmbedder {
    core: DeterministicEmbedderCore,
    endpoint: String,
    auth_token: Option<String>,
}

impl OnnxHttpEmbedder {
    pub fn new(config: OnnxHttpConfig) -> Result<Self, EmbedderError> {
        let info = EmbedderInfo {
            provider: ProviderKind::OnnxHttp,
            embedding_model_id: config.embedding_model_id.clone(),
            dimension: config.dimension,
            text_repr_version: config.text_repr_version.clone(),
        };

        let unique = match &config.auth_token {
            Some(token) => format!("{}::{}", config.endpoint, token),
            None => config.endpoint.clone(),
        };

        let base_seed = compute_seed(
            ProviderKind::OnnxHttp,
            &unique,
            &config.embedding_model_id,
            &config.text_repr_version,
        );

        let core = DeterministicEmbedderCore::new(info, config.max_input_length, base_seed)?;

        Ok(Self {
            core,
            endpoint: config.endpoint,
            auth_token: config.auth_token,
        })
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn auth_token(&self) -> Option<&str> {
        self.auth_token.as_deref()
    }
}

impl Embedder for OnnxHttpEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        self.core.embed(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        self.core.embed_batch(texts)
    }

    fn info(&self) -> &EmbedderInfo {
        self.core.info()
    }
}

#[derive(Debug, Clone)]
struct DeterministicEmbedderCore {
    info: EmbedderInfo,
    max_input_length: usize,
    base_seed: u64,
}

impl DeterministicEmbedderCore {
    fn new(
        info: EmbedderInfo,
        max_input_length: usize,
        base_seed: u64,
    ) -> Result<Self, EmbedderError> {
        if info.dimension == 0 {
            return Err(EmbedderError::InvalidConfiguration {
                message: "dimension must be greater than zero".into(),
            });
        }

        if max_input_length == 0 {
            return Err(EmbedderError::InvalidConfiguration {
                message: "max_input_length must be greater than zero".into(),
            });
        }

        Ok(Self {
            info,
            max_input_length,
            base_seed,
        })
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        self.validate_length(text)?;
        Ok(self.generate_embedding(text))
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedderError> {
        texts
            .iter()
            .map(|text| self.embed(text))
            .collect::<Result<Vec<_>, _>>()
    }

    fn info(&self) -> &EmbedderInfo {
        &self.info
    }

    fn validate_length(&self, text: &str) -> Result<(), EmbedderError> {
        let actual_length = text.chars().count();
        if actual_length > self.max_input_length {
            return Err(EmbedderError::InputTooLong {
                max_length: self.max_input_length,
                actual_length,
            });
        }
        Ok(())
    }

    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.info.dimension);
        for index in 0..self.info.dimension {
            let mut hasher = DefaultHasher::new();
            self.base_seed.hash(&mut hasher);
            index.hash(&mut hasher);
            text.hash(&mut hasher);
            let hash = hasher.finish();
            output.push(normalize_hash(hash));
        }
        output
    }
}

fn ensure_ort_initialized(runtime_library_path: &Path) -> Result<(), EmbedderError> {
    if let Some(existing) = ORT_RUNTIME_PATH.get() {
        if !paths_equal(existing, runtime_library_path) {
            return Err(EmbedderError::InvalidConfiguration {
                message: format!(
                    "ONNX Runtime already initialized with library `{}`; cannot reinitialize with `{}`",
                    existing.display(),
                    runtime_library_path.display()
                ),
            });
        }
    } else {
        let _ = ORT_RUNTIME_PATH.set(runtime_library_path.to_path_buf());
    }

    ort::init_from(runtime_library_path.to_string_lossy().to_string())
        .with_name("hybred-search")
        .commit()
        .map_err(|err| map_ort_error("initialize ONNX Runtime environment", err))?;

    Ok(())
}

fn resolve_existing_path(path: &Path, description: &str) -> Result<PathBuf, EmbedderError> {
    fs::metadata(path).map_err(|_| EmbedderError::InvalidConfiguration {
        message: format!("{description} `{}` does not exist", path.display()),
    })?;

    path.canonicalize()
        .map_err(|err| EmbedderError::ProviderFailure {
            message: format!(
                "failed to canonicalize {description} `{}`: {err}",
                path.display()
            ),
        })
}

fn map_ort_error(context: &str, err: OrtError) -> EmbedderError {
    EmbedderError::ProviderFailure {
        message: format!("{context} failed: {err}"),
    }
}

fn map_tokenizer_error(context: &str, err: tokenizers::Error) -> EmbedderError {
    EmbedderError::ProviderFailure {
        message: format!("{context} failed: {err}"),
    }
}

// Removed unused `map_shape_error`; all ndarray constructions are infallible in current code.

fn compute_seed(
    provider: ProviderKind,
    unique: &str,
    embedding_model_id: &str,
    text_repr_version: &str,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    provider.hash(&mut hasher);
    unique.hash(&mut hasher);
    embedding_model_id.hash(&mut hasher);
    text_repr_version.hash(&mut hasher);
    hasher.finish()
}

fn normalize_hash(value: u64) -> f32 {
    const SCALE: f64 = 2.0;
    let normalized = (value as f64) / (u64::MAX as f64);
    (normalized * SCALE - 1.0) as f32
}

fn paths_equal(a: &Path, b: &Path) -> bool {
    if let (Ok(a), Ok(b)) = (fs::canonicalize(a), fs::canonicalize(b)) {
        a == b
    } else {
        a == b
    }
}
