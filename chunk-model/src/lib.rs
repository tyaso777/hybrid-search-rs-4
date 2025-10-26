//! Shared, lightweight chunk schema and helpers.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Semantic version of the NDJSON/JSON record schema (major bumps are breaking).
pub const SCHEMA_MAJOR: u16 = 1;
pub const SCHEMA_MINOR: u16 = 0;

/// Opaque document identifier. String keeps it flexible (UUID/ULID/hash).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocumentId(pub String);

/// Opaque chunk identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId(pub String);

/// Logical section path like ["Ⅰ 基本的考え方", "Ⅹ－５ 補償"].
pub type SectionPath = Vec<String>;

/// One NDJSON line = one chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkRecord {
    /// Schema version (store major only for tolerant readers).
    pub schema_version: u16,
    /// Source document identifier.
    pub doc_id: DocumentId,
    /// Unique id for this chunk.
    pub chunk_id: ChunkId,
    /// Source URI (file://, s3://, etc.).
    pub source_uri: String,
    /// Source MIME/type string (e.g., "application/pdf").
    pub source_mime: String,
    /// Extraction timestamp in ISO 8601 (UTC). Optional producers may leave empty.
    pub extracted_at: String,
    /// Searchable text body.
    pub text: String,
    /// Optional logical section path within the document.
    pub section_path: SectionPath,
    /// Lightweight metadata bag.
    pub meta: BTreeMap<String, String>,

    /// Forward-compatible extension area. Namespaced keys recommended (e.g., "layout.span").
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ChunkRecord {
    /// Soft validation suitable for ingestion.
    pub fn validate_soft(&self) -> Result<(), String> {
        if self.text.trim().is_empty() {
            return Err("text is empty".into());
        }
        Ok(())
    }
}

