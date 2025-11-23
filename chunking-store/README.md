## Chunking Store

[← Back to workspace README](../README.md)

Storage and indexing helpers for `chunk_model::ChunkRecord` (e.g., SQLite/Tantivy/HNSW).

### What It Provides (current)
- Thin abstractions
  - `ChunkPrimaryStore` (DB-agnostic primary store)
  - `SearchHit` (unified result item)
- SQLite-backed `SqliteRepo` (primary store)
- Stub modules for `tantivy_index` and `hnsw_index` (to be integrated next)

### Status
- SQLite persistence implemented in `SqliteRepo`.
- Vector indexing (HNSW) and Tantivy integration are planned next.

### Example (trait usage)
```rust
use chunking_store::sqlite_repo::SqliteRepo;
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
    section_path: None,
    meta: BTreeMap::new(),
    extra: BTreeMap::new(),
};

repo.upsert_chunks(vec![rec]).unwrap();

let loaded = repo.get_chunks_by_ids(&[ChunkId("doc-001#0".into())]).unwrap();
assert_eq!(loaded.len(), 1);
```

#### Notes
- `upsert_chunks` wraps a single SQLite transaction (`BEGIN IMMEDIATE`) for atomicity in the main store.
- This phase does not create Tantivy/HNSW entries yet; those will be queued in a later phase.

### CLI Examples and Default DB Location

- We provide small examples to validate ingestion and deletion flows:

  - Ingest demo (SQLite primary store only):
    - Default DB path: `target/demo/chunks.db`
    - Commands:
      - `cargo run -p chunking-store --example ingest_demo -- --sample`
      - `cargo run -p chunking-store --example ingest_demo -- ./my.db --ndjson ./chunks.ndjson`
    - Flags:
      - `--sample` inserts two demo chunks (EN/JA)
      - `--ndjson PATH` ingests NDJSON of `ChunkRecord`

  - Delete by filters (DB → indexes orchestrated):
    - Default DB path: `target/demo/chunks.db`
    - Commands:
      - `cargo run -p chunking-store --example delete_demo -- --doc-id doc-001`
      - `cargo run -p chunking-store --example delete_demo -- ./my.db --prefix file:///data/ --start 2024-01-01T00:00:00Z --end 2025-01-01T00:00:00Z`

- Housekeeping
  - `.gitignore` contains `*.db` so ad-hoc DBs are not tracked.
  - A helper script to clean all `.db` files exists: `scripts/clean_dbs.ps1` (PowerShell)
    - Run: `pwsh scripts/clean_dbs.ps1` (use `-Force` to skip prompt)
