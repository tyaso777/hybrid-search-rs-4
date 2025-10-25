use chunk_model::ChunkRecord;

/// Stub FTS5 index wrapper.
#[derive(Default)]
pub struct Fts5Index;

impl Fts5Index {
    pub fn new() -> Self { Self }
    pub fn index(&mut self, _chunks: &[ChunkRecord]) {
        // no-op stub
    }
}

