## Chunking Store

[← Back to workspace README](../README.md)

Storage and indexing helpers for `chunk_model::ChunkRecord` (e.g., SQLite/FTS5/Tantivy/HNSW).

### What It Provides (current)
- `ChunkStore` trait with `insert_chunks` and `search`
- Stub modules: `sqlite_repo`, `fts5_index`, `tantivy_index`, `hnsw_index`
- In-memory data storage in `sqlite_repo` (stub implementation)

### Status
- Scaffolding only. Real persistence, FTS/vector indexing, and queries are to be implemented.

### Example (trait usage)
```rust
use chunking_store::{ChunkStore, sqlite_repo::SqliteRepo};
use chunk_model::{ChunkRecord, DocumentId, ChunkId, SCHEMA_MAJOR};
use std::collections::BTreeMap;

let mut repo = SqliteRepo::new();

let rec = ChunkRecord {
    schema_version: SCHEMA_MAJOR,
    doc_id: DocumentId("doc-001".into()),
    chunk_id: ChunkId("doc-001#0".into()),
    source_uri: "./docs/sample.txt".into(),
    source_mime: "text/plain".into(),
    extracted_at: "".into(),
    text: "hello".into(),
    section_path: vec![],
    meta: BTreeMap::new(),
    extra: BTreeMap::new(),
};

repo.insert_chunks(vec![rec]);
let _hits = repo.search("hello", 10);
```

