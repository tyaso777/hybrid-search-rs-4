# hybrid-search-rs-4

Rust workspace for a hybrid search pipeline.

- file-chunker: read & chunk files (PDF/DOCX stubs)
- chunking-store: store & search (SQLite/FTS5/Tantivy/HNSW stubs)
- chunk-model: shared data models

Scaffolded and buildable. Extend each crate with real implementations.

## Build

```
cargo build
```

