## Hybrid Orchestrator (SQLite + HNSW)

[← Back to workspace README](../../README.md)

> Note
> This CLI is kept for reference and basic testing. For most workflows, prefer the desktop GUI: `tools/hybrid-orchestrator-gui`. You likely don't need to use this CLI in day-to-day work.

Command-line tool to ingest plain text and search over the vector index:
- Primary store: SQLite (`chunks` table) via `chunking-store::sqlite_repo::SqliteRepo`
- Vector search: HNSW (cosine) persisted as a snapshot directory
- Embeddings: local ONNX model via `embedding_provider`

### Build

```
cargo build -p hybrid-orchestrator
```

### Usage

```
hybrid-orchestrator insert [db_path] --text TEXT [--doc DOC] [--hnsw DIR]
hybrid-orchestrator insert [db_path] --stdin       [--doc DOC] [--hnsw DIR]

hybrid-orchestrator search [db_path] --query Q [--k N] [--hnsw DIR]

# Optional model overrides (insert or search):
  --model PATH_ONNX   --tokenizer PATH_JSON   --runtime PATH_DLL   --dim N   --max-tokens N

# Defaults
- db_path: target/demo/chunks.db
- hnsw:    <db_path>.hnsw
- dim/max tokens: from `embedding_provider::config::ONNX_STDIO_DEFAULTS`
```

### Examples

```
# Ingest a single text
cargo run -p hybrid-orchestrator -- insert --text "こんにちは Rust! ベクトル検索も試します。"

# Vector search
cargo run -p hybrid-orchestrator -- search --query Rust

# Using custom model/tokenizer/runtime
cargo run -p hybrid-orchestrator -- insert ./my.db --text "custom paths" \
  --model path/to/model.onnx \
  --tokenizer path/to/tokenizer.json \
  --runtime path/to/onnxruntime.dll \
  --dim 768 --max-tokens 8192
```

### Notes
- HNSW snapshot is stored under `--hnsw` (default: `<db_path>.hnsw`). It is loaded on `insert` and `search`.
- Embedding requires the ONNX model, tokenizer, and ONNX Runtime DLL. Defaults come from `embedding_provider/src/config.rs` and resolve relative to that crate.
