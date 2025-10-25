//! Shared models used across crates

/// A single chunk of text derived from source content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkRecord {
    /// Unique id for this chunk within a corpus or store.
    pub id: String,
    /// Source document metadata.
    pub source: SourceMeta,
    /// Stable order of the chunk within the source.
    pub order: u32,
    /// Text content of the chunk.
    pub text: String,
    /// Estimated token count (optional; can be 0 if unknown).
    pub tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceMeta {
    /// URI or file path string for the source document.
    pub uri: String,
    /// MIME-like content type (e.g., "application/pdf").
    pub content_type: String,
}

impl SourceMeta {
    pub fn new(uri: impl Into<String>, content_type: impl Into<String>) -> Self {
        Self { uri: uri.into(), content_type: content_type.into() }
    }
}

impl ChunkRecord {
    pub fn new(id: impl Into<String>, source: SourceMeta, order: u32, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            source,
            order,
            text: text.into(),
            tokens: 0,
        }
    }
}

