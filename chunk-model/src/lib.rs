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

/// Minimal inline link representation within a chunk's text body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkRef {
    pub start: u32,
    pub end: u32,
    pub uri: String,
}

/// Content type for downstream retrieval & rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockKind {
    Heading,      // chapters/sections/subsections
    Paragraph,    // prose
    ListItem,     // bullet/numbered items
    Table,        // structured table
    Figure,       // embedded or referenced image
    Caption,      // figure/table caption
    Code,         // code or pseudocode
    Equation,     // math equations
    Footnote,     // footnotes
    HeaderFooter, // page header/footer (usually not indexed)
}

/// File-level metadata shared by all chunks of a document.
/// Keep document-scoped attributes here and join to chunks via `doc_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRecord {
    /// Schema version (store major only for tolerant readers).
    pub schema_version: u16,
    /// Logical document identifier. Stable across chunks and revisions of the same logical doc.
    pub doc_id: DocumentId,
    /// Optional revision counter for re-ingestion/versioning (1,2,3,...).
    pub doc_revision: Option<u32>,

    /// Origin information.
    pub source_uri: String,
    pub source_mime: String,

    /// File facts (if available).
    pub file_size_bytes: Option<u64>,
    pub content_sha256: Option<String>,
    pub page_count: Option<u32>,

    /// Timestamps as ISO 8601 (UTC) strings to avoid hard deps.
    pub extracted_at: String,
    pub created_at_meta: Option<String>,
    pub updated_at_meta: Option<String>,

    /// Doc-level guesses/labels.
    pub title_guess: Option<String>,
    pub author_guess: Option<String>,
    pub dominant_lang: Option<String>,
    pub tags: Vec<String>,

    /// Ingestion process info (for reproducibility/audit).
    pub ingest_tool: Option<String>,
    pub ingest_tool_version: Option<String>,
    pub reader_backend: Option<String>,   // e.g., "pdfium", "pure-pdf"
    pub ocr_used: Option<bool>,
    pub ocr_langs: Vec<String>,

    /// Aggregates (optional, can be computed on read path too).
    pub chunk_count: Option<u32>,
    pub total_tokens: Option<u32>,

    /// Lightweight metadata and forward-compatible extension area.
    pub meta: BTreeMap<String, String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

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

