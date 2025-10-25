use chunk_model::ChunkRecord;

/// Stub HNSW index wrapper.
#[derive(Default)]
pub struct HnswIndex;

impl HnswIndex {
    pub fn new() -> Self { Self }
    pub fn index(&mut self, _chunks: &[ChunkRecord]) {
        // no-op stub
    }
}

