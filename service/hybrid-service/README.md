Hybrid Service

Overview
- Provides a cohesive service layer that ties together:
  - Chunking (PDF/DOCX) via `file-chunker`
  - Primary storage (SQLite + FTS5) via `chunking-store`
  - Vector index (HNSW) via `chunking-store`
  - Embeddings via `embedding-provider`

Key APIs
- `HybridService::ingest_file(path, doc_id_hint)`
- `HybridService::ingest_chunks(records, vectors)`
- `HybridService::search_text(query, top_k, filters)`
- `HybridService::search_hybrid(query, top_k, filters, w_text, w_vec)`
- `HybridService::delete_by_filter(filters, batch_size)`
- `HybridService::repo_counts()`

Quick Start
1) Build: `cargo build -p hybrid-service`
2) Example: `cargo run -p hybrid-service --example ingest_and_search -- <FILE> <QUERY>`

Notes
- Defaults store data under `target/demo/chunks.db` and `target/demo/chunks.db.hnsw`.
- The embedder uses `embedding_provider::config::default_stdio_config()`; override in `ServiceConfig` if needed.

