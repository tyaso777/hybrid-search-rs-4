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
- FTS5 text search provided via `Fts5Index` (uses SQLite triggers for maintenance). Note: ranking via `bm25()` can be unavailable depending on the SQLite build; in some environments, MATCH queries may return 0 from our pipeline even though the raw FTS table matches. Treat FTS5 integration as WIP for ranking/compat, and prefer Tantivy for production relevance ranking.
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
    section_path: None,
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

### FTS5 Caveats (WIP)
- Some SQLite builds expose `fts5` but not the `bm25()` function; ordering by `bm25` may fail or behave inconsistently.
- Depending on tokenizer and build options, our current `Fts5Index` pipeline can report 0 hits while a direct `SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'q'` shows matches. This is being tracked; use Tantivy for robust text retrieval and relevance ranking for now.
- To debug locally, the `ingest_demo` example supports `--debug` to print `chunks`/`chunks_fts` row counts and a direct `MATCH` count.

### CLI Examples and Default DB Location

- We provide small examples to validate ingestion and deletion flows:

  - Ingest + search (FTS5, English-friendly):
    - Default DB path: `target/demo/chunks.db`
    - Commands:
      - `cargo run -p chunking-store --example ingest_demo -- --sample --search hello`
      - `cargo run -p chunking-store --example ingest_demo -- ./my.db --ndjson ./chunks.ndjson --search "your query"`
    - Flags:
      - `--sample` inserts two demo chunks (EN/JA)
      - `--ndjson PATH` ingests NDJSON of `ChunkRecord`
      - `--search QUERY` runs a simple FTS5 search
      - `--debug` prints `chunks`/`chunks_fts` counts and raw `MATCH` diagnostics

  - Delete by filters (DB → indexes orchestrated):
    - Default DB path: `target/demo/chunks.db`
    - Commands:
      - `cargo run -p chunking-store --example delete_demo -- --doc-id doc-001`
      - `cargo run -p chunking-store --example delete_demo -- ./my.db --prefix file:///data/ --start 2024-01-01T00:00:00Z --end 2025-01-01T00:00:00Z`

- Housekeeping
  - `.gitignore` contains `*.db` so ad-hoc DBs are not tracked.
  - A helper script to clean all `.db` files exists: `scripts/clean_dbs.ps1` (PowerShell)
    - Run: `pwsh scripts/clean_dbs.ps1` (use `-Force` to skip prompt)
