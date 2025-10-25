use chunk_model::ChunkRecord;
use crate::ChunkStore;

/// Stubbed SQLite-backed repository (in-memory for now).
#[derive(Default)]
pub struct SqliteRepo {
    data: Vec<ChunkRecord>,
}

impl SqliteRepo {
    pub fn new() -> Self { Self { data: Vec::new() } }
}

impl ChunkStore for SqliteRepo {
    fn insert_chunks(&mut self, chunks: Vec<ChunkRecord>) {
        self.data.extend(chunks);
    }

    fn search(&self, _query: &str, limit: usize) -> Vec<ChunkRecord> {
        // naive: return first N inserted chunks
        self.data.iter().cloned().take(limit).collect()
    }
}

