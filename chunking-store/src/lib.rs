pub mod sqlite_repo;
pub mod fts5_index;
pub mod tantivy_index;
pub mod hnsw_index;

use chunk_model::ChunkRecord;

/// Thin abstraction for the primary storage engine (DB-agnostic).
pub trait ChunkPrimaryStore {
    /// Upsert a batch of chunks atomically within the store.
    fn upsert_chunks(&mut self, chunks: Vec<ChunkRecord>) -> Result<(), StoreError>;
}

/// Placeholder for text search engines (FTS5/Tantivy, etc.).
/// Concrete engines will expose their own constructors; common querying shape is unified via helpers.
#[derive(Debug, Clone)]
pub struct SearchHit {
    pub chunk: ChunkRecord,
    pub score: f32,
}

/// Store-agnostic text match result (IDs only). Useful for composing with any primary store.
#[derive(Debug, Clone, PartialEq)]
pub struct TextMatch {
    pub chunk_id: chunk_model::ChunkId,
    pub score: f32,
}

/// Optional read-side abstraction so indexes can fetch full records in a store-agnostic way.
pub trait ChunkStoreRead {
    fn get_chunks_by_ids(
        &self,
        ids: &[chunk_model::ChunkId],
    ) -> Result<Vec<ChunkRecord>, StoreError>;
    fn as_any(&self) -> &dyn std::any::Any;
}

// ------------------------------
// Filters and query options
// ------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum FilterOp {
    DocIdEq(String),
    DocIdIn(Vec<String>),
    SourceUriPrefix(String),
    MetaEq { key: String, value: String },
    MetaIn { key: String, values: Vec<String> },
    /// Numeric range on a field (e.g., meta value). Missing/parse-failed values do not match.
    RangeNumeric { key: String, min: Option<f64>, max: Option<f64>, min_incl: bool, max_incl: bool },
    /// ISO 8601 string range (lexicographic compare). Works for fields like `extracted_at` or ISO dates in meta.
    RangeIsoDate { key: String, start: Option<String>, end: Option<String>, start_incl: bool, end_incl: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterKind {
    Must,
    PreferPre,
    PostOnly,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FilterClause {
    pub kind: FilterKind,
    pub op: FilterOp,
}

#[derive(Debug, Clone, Copy)]
pub struct IndexCaps {
    pub can_prefilter_doc_id_eq: bool,
    pub can_prefilter_doc_id_in: bool,
    pub can_prefilter_source_prefix: bool,
    pub can_prefilter_meta: bool,
    pub can_prefilter_range_numeric: bool,
    pub can_prefilter_range_date: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct SearchOptions {
    pub top_k: usize,
    pub fetch_factor: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions { top_k: 10, fetch_factor: 10 }
    }
}

pub trait TextSearcher {
    fn name(&self) -> &'static str;
    fn caps(&self) -> IndexCaps;
    fn search_ids(
        &self,
        store: &dyn ChunkStoreRead,
        query: &str,
        filters: &[FilterClause],
        opts: &SearchOptions,
    ) -> Vec<TextMatch>;
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("backend error: {0}")]
    Backend(String),
}

