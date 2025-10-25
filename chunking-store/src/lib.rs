pub mod sqlite_repo;
pub mod fts5_index;
pub mod tantivy_index;
pub mod hnsw_index;

use chunk_model::ChunkRecord;

/// Basic trait representing a chunk store and search capability.
pub trait ChunkStore {
    fn insert_chunks(&mut self, chunks: Vec<ChunkRecord>);
    fn search(&self, _query: &str, limit: usize) -> Vec<ChunkRecord>;
}

