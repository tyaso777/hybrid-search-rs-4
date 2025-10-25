use chunk_model::ChunkRecord;

/// Stub Tantivy index wrapper.
#[derive(Default)]
pub struct TantivyIndex;

impl TantivyIndex {
    pub fn new() -> Self { Self }
    pub fn index(&mut self, _chunks: &[ChunkRecord]) {
        // no-op stub
    }
}

