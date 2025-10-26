## Chunking Store

[← Back to workspace README](../README.md)

Storage and indexing helpers for `chunk_model::ChunkRecord` (e.g., SQLite/FTS5/Tantivy/HNSW).

### What It Provides (current)
- Thin abstractions
  - `ChunkPrimaryStore` (DB-agnostic primary store)
  - `SearchHit` (unified result item)
- SQLite-backed `SqliteRepo` (primary store) + FTS5 index in `fts5_index` (search)
- Stub modules for `tantivy_index` and `hnsw_index` (to be integrated next)

### Status
- SQLite persistence implemented in `SqliteRepo`.
- FTS5 text search provided via `Fts5Index` (uses SQLite triggers for maintenance).
- Vector indexing (HNSW) and Tantivy integration are planned next.

### Example (trait usage)
```rust
use chunking_store::sqlite_repo::SqliteRepo;
use chunking_store::fts5_index::Fts5Index;
use chunking_store::ChunkPrimaryStore;
use chunk_model::{ChunkRecord, DocumentId, ChunkId, SCHEMA_MAJOR};
use std::collections::BTreeMap;

// In-memory DB. Use `SqliteRepo::open("path/to.db")` for file-backed.
let mut repo = SqliteRepo::new();

let rec = ChunkRecord {
    schema_version: SCHEMA_MAJOR,
    doc_id: DocumentId("doc-001".into()),
    chunk_id: ChunkId("doc-001#0".into()),
    source_uri: "./docs/sample.txt".into(),
    source_mime: "text/plain".into(),
    extracted_at: "".into(),
    text: "hello world".into(),
    section_path: vec![],
    meta: BTreeMap::new(),
    extra: BTreeMap::new(),
};

repo.upsert_chunks(vec![rec]).unwrap();

let fts = Fts5Index::new();
let hits = fts.search_simple(&repo, "hello", 10);
assert!(!hits.is_empty());
```

#### Notes
- FTS5 is maintained via triggers on `chunks` for insert/update/delete.
- `upsert_chunks` wraps a single SQLite transaction (`BEGIN IMMEDIATE`) for atomicity in the main store.
- This phase does not create Tantivy/HNSW entries yet; those will be queued in a later phase.
