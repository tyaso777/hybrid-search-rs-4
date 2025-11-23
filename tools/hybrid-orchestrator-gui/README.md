## Hybrid Orchestrator GUI

[← Back to workspace README](../../README.md)

Simple desktop GUI to ingest plain text into SQLite + HNSW (with optional Tantivy text search) and run text or hybrid search.

### Build/Run

```
cargo run -p hybrid-orchestrator-gui
```

### Features
- Configure ONNX model, tokenizer, and ONNX Runtime DLL (Windows)
- Configure SQLite DB path, HNSW snapshot directory, and Tantivy index directory (when the `tantivy` feature is enabled)
- Insert: enter text and optional doc ID, auto-generates metadata and vector, updates DB/HNSW (and Tantivy when enabled)
- Search: text-only (Tantivy) or hybrid (Tantivy + vector) with simple weighted fusion

Notes
- Embedding defaults are resolved relative to `embedding_provider`; adjust in the UI or via `embedding_provider/src/config.rs`.
- Tantivy BM25 scoring follows its default schema; use AND/OR tokenization controls in the UI to tweak matching.
