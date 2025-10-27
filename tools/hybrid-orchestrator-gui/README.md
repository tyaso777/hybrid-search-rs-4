## Hybrid Orchestrator GUI

[← Back to workspace README](../../README.md)

Simple desktop GUI to ingest plain text into SQLite + FTS5 + HNSW and run text or hybrid search.

### Build/Run

```
cargo run -p hybrid-orchestrator-gui
```

### Features
- Configure ONNX model, tokenizer, and ONNX Runtime DLL (Windows)
- Configure SQLite DB path and HNSW snapshot directory
- Insert: enter text and optional doc ID, auto-generates metadata and vector, updates DB/FTS/HNSW
- Search: text-only (FTS5) or hybrid (FTS5 + vector) with simple weighted fusion

Notes
- Embedding defaults are resolved relative to `embedding_provider`; adjust in the UI or via `embedding_provider/src/config.rs`.
- FTS5 ranking via `bm25()` depends on your SQLite build.
