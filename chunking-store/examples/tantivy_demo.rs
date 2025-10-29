#![cfg(feature = "tantivy-impl")]

use chunk_model::{ChunkId, ChunkRecord, DocumentId, SCHEMA_MAJOR};
use chunking_store::tantivy_index::TantivyIndex;
use chunking_store::{ChunkStoreRead, FilterClause, FilterKind, FilterOp, SearchOptions, StoreError, TextSearcher};
use std::collections::BTreeMap;

struct NullStore;

impl ChunkStoreRead for NullStore {
    fn get_chunks_by_ids(&self, _ids: &[ChunkId]) -> Result<Vec<ChunkRecord>, StoreError> {
        Ok(Vec::new())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn make_rec(id: &str, text: &str, extracted_at: &str) -> ChunkRecord {
    ChunkRecord {
        schema_version: SCHEMA_MAJOR,
        doc_id: DocumentId("doc-001".into()),
        chunk_id: ChunkId(id.into()),
        source_uri: "memory://demo".into(),
        source_mime: "text/plain".into(),
        extracted_at: extracted_at.into(),
        text: text.into(),
        section_path: None,
        meta: BTreeMap::new(),
        extra: BTreeMap::new(),
    }
}

fn main() {
    // Build an in-memory Tantivy index
    let idx = TantivyIndex::new_ram().expect("init tantivy index");

    // Upsert a few records
    let recs = vec![
        make_rec("c1", "hello world", "2024-01-02T00:00:00Z"),
        make_rec("c2", "greetings earth", "2024-06-01T00:00:00Z"),
        make_rec("c3", "hello rust", "2025-01-10T00:00:00Z"),
    ];
    idx.upsert_records(&recs).expect("upsert records");

    // Search with filters: doc_id eq and date range within 2024
    let filters = vec![
        FilterClause { kind: FilterKind::Must, op: FilterOp::DocIdEq("doc-001".into()) },
        FilterClause { kind: FilterKind::PreferPre, op: FilterOp::RangeIsoDate {
            key: "extracted_at".into(),
            start: Some("2024-01-01T00:00:00Z".into()),
            end: Some("2025-01-01T00:00:00Z".into()),
            start_incl: true,
            end_incl: false,
        }},
    ];

    let opts = SearchOptions { top_k: 5, fetch_factor: 5 };
    let store = NullStore;
    let hits = idx.search_ids(&store, "hello", &filters, &opts);

    println!("hits: {}", hits.len());
    for h in hits {
        println!("id={} score={:.4} raw={:.4}", (h.chunk_id).0, h.score, h.raw_score);
    }
    // Done
}
