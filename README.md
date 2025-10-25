# hybrid-search-rs-4

Rust workspace for a hybrid search pipeline.

## Workspace Crates

- file-chunker
  - Utilities to split files into content chunks for downstream indexing and retrieval.

- chunk-model
  - Shared data types and traits used across the workspace (chunk metadata, content representations, etc.).

- chunking-store
  - Storage layer for chunks (e.g., SQLite-backed repository) and related persistence helpers.

- embedding_provider
  - Local ONNX-based embedding library (ONNX Runtime + tokenizers). Produces text embeddings via masked mean pooling.
  - Docs: see [embedding_provider/README.md](embedding_provider/README.md) for model preparation, configuration defaults, and tests.

- tools/embedder-demo
  - Desktop GUI (egui/eframe) for interactive embedding. See usage in [embedding_provider/README.md](embedding_provider/README.md).

## Documentation Map

- Embedding Provider Guide
  - Path: [embedding_provider/README.md](embedding_provider/README.md)
  - Read this if you want to set up a local ONNX embedding model, export an ONNX from a HF model, understand pooling/dimension checks, or run the CLI/GUI with the right defaults.

- Demo Test Data
  - Path: [tools/embedder-demo/testdata/README.md](tools/embedder-demo/testdata/README.md)
  - Read this if you want to know where to place sample Excel/CSV files for the GUI demo, which files are committed (small) vs ignored (large), and where outputs are written.

- Workspace Overview (this page)
  - Read this to understand the overall crate structure, how they relate, and quick commands to build/test/run.

## Quick Start

- Build everything: `cargo build`
- Run tests for embeddings: `cargo test -p embedding-provider`
- CLI sanity check: `cargo run -p embedding-provider --bin embed_cli "your text"`

- file-chunker: read & chunk files (PDF/DOCX stubs)
- chunking-store: store & search (SQLite/FTS5/Tantivy/HNSW stubs)
- chunk-model: shared data models

Scaffolded and buildable. Extend each crate with real implementations.

## Build

```
cargo build
```

