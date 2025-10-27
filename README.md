# hybrid-search-rs-4

Rust workspace for a hybrid search pipeline.

## Workspace Crates

- file-chunker
  - Utilities to split files into content chunks for downstream indexing and retrieval. See [file-chunker/README.md](file-chunker/README.md).

- chunk-model
  - Shared data types and traits used across the workspace (chunk metadata, content representations, etc.). See [chunk-model/README.md](chunk-model/README.md).

- chunking-store
  - Storage layer for chunks (e.g., SQLite-backed repository) and related persistence helpers. See [chunking-store/README.md](chunking-store/README.md).

- embedding_provider
  - Local ONNX-based embedding library (ONNX Runtime + tokenizers). Produces text embeddings via masked mean pooling.
  - Docs: see [embedding_provider/README.md](embedding_provider/README.md) for model preparation, configuration defaults, and tests.
  - Model loading modes: direct-from-file (default) or preload-to-memory via `OnnxStdIoConfig { preload_model_to_memory: true }` for slow network shares.

- tools/embedder-demo
  - Desktop GUI (egui/eframe) for interactive embedding. See usage in [embedding_provider/README.md](embedding_provider/README.md).
  - Includes a checkbox to preload the model into memory when initializing.

- tools/hybrid-orchestrator
  - CLI for ingest/search with SQLite + FTS5 + HNSW.
  - Status: legacy/optional. For most workflows, prefer the GUI at `tools/hybrid-orchestrator-gui`. You likely don't need this CLI.
  - Docs: see [tools/hybrid-orchestrator/README.md](tools/hybrid-orchestrator/README.md).

- tools/hybrid-orchestrator-gui
  - Desktop GUI for end-to-end ingest/search (SQLite + FTS5 + Tantivy + HNSW). Also supports the model preload option in the UI.
  - Docs: see [tools/hybrid-orchestrator-gui/README.md](tools/hybrid-orchestrator-gui/README.md).

- tools/pdf-block-viewer
  - GUI to inspect PDF extraction results (UnifiedBlocks) from `file-chunker` with multiple backends (stub/pure-rust/pdfium).

- tools/tokenize-lab
  - GUI sandbox for tokenization experiments (Lindera, Tantivy QueryParser analyzer, N-gram).

## Documentation Map

- Embedding Provider Guide
  - Path: [embedding_provider/README.md](embedding_provider/README.md)
  - Read this if you want to set up a local ONNX embedding model, export an ONNX from a HF model, understand pooling/dimension checks, or run the CLI/GUI with the right defaults.

- Models Folder Guide
  - Path: [embedding_provider/models/README.md](embedding_provider/models/README.md)
  - Read this to see where model.onnx and tokenizer assets should live and why they are not committed.

- Runtime Binaries Guide
  - Path: [embedding_provider/bin/README.md](embedding_provider/bin/README.md)
  - Read this to see where ONNX Runtime DLLs should be placed and why they are ignored by Git.

- Demo Test Data
  - Path: [tools/embedder-demo/testdata/README.md](tools/embedder-demo/testdata/README.md)
  - Read this if you want to know where to place sample Excel/CSV files for the GUI demo, which files are committed (small) vs ignored (large), and where outputs are written.

- Workspace Overview (this page)
  - Read this to understand the overall crate structure, how they relate, and quick commands to build/test/run.

- Chunk Model Guide
  - Path: [chunk-model/README.md](chunk-model/README.md)
  - Read this to understand the shared chunk schema, versioning policy, and usage examples.

- File Chunker Guide
  - Path: [file-chunker/README.md](file-chunker/README.md)
  - Read this to understand the chunking pipeline, current stubs, and output shape.

- Chunking Store Guide
  - Path: [chunking-store/README.md](chunking-store/README.md)
  - Read this to understand the store/index abstractions and current stubs.

## Quick Start

0) Prepare ONNX Runtime and Model (one‑time)
- Follow the setup guide in [embedding_provider/README.md](embedding_provider/README.md) to place the ONNX Runtime shared library and the ONNX model/tokenizer.
- If you use non‑default locations, either edit `embedding_provider/src/config.rs` or set the paths in the GUI fields when you run the tools.

1) Build everything: `cargo build`
2) Optional tests: `cargo test -p embedding-provider`
3) Sanity check (CLI): `cargo run -p embedding-provider --bin embed_cli "your text"`
4) Try the Hybrid Orchestrator GUI:
   - `cargo run -p hybrid-orchestrator-gui`
   - Configure model/tokenizer/runtime if you didn't use the defaults above; then Insert or Excel Ingest, and Search.

- file-chunker: read & chunk files (PDF/DOCX stubs)
- chunking-store: store & search (SQLite/FTS5/Tantivy/HNSW stubs)
- chunk-model: shared data models

Scaffolded and buildable. Extend each crate with real implementations.

## Compliance / Security

- Project license
  - MIT. See [LICENSE](LICENSE).

- Security policy
  - See [SECURITY.md](SECURITY.md) for reporting guidance and monitoring tools.

- Third‑party licenses (for distribution)
  - Summary: `reports/THIRD-PARTY-NOTICES.txt`
  - Full texts: `reports/THIRD-PARTY-LICENSES.txt` (generated via cargo-about)
  - Include these files with distributable artifacts when you ship binaries.
  - Details and checklist: see [docs/compliance.md](docs/compliance.md)

- Regenerate license and vulnerability reports
  - Windows (PowerShell): `powershell -ExecutionPolicy Bypass -File scripts/generate_reports.ps1`
  - Bash: `bash scripts/generate_reports.sh`
  - Outputs: `reports/cargo-audit.*`, `reports/license.*`, `reports/THIRD-PARTY-*`

- Continuous checks (CI)
  - GitHub Actions workflow runs `cargo deny check` on push/PR to `main`.
  - Config: `.github/workflows/cargo-deny.yml`, policy: `deny.toml`

- Local checks
  - `cargo deny check`
  - Optional: `cargo audit` for a RustSec scan, `cargo about generate` for license data

## Build

```
cargo build
```

## Hybrid Orchestrator GUI

Interactive end-to-end tool to ingest and search with SQLite + FTS5 + Tantivy + HNSW.

- Run
  - `cargo run -p hybrid-orchestrator-gui`

- Configure (top of the window)
  - Embedding model (ONNX), tokenizer JSON, and ONNX Runtime DLL (paths default to `embedding_provider` config)
  - SQLite DB path (default: `target/demo/chunks.db`)
  - HNSW Dir (default: `<db>.hnsw`)
  - Tantivy Dir (default: `<db>.tantivy`)

- Tabs
  - Insert: type text and insert a single chunk (vector is generated automatically)
  - Excel Ingest: pick a 1-column workbook and ingest each row as a chunk (batch embedding)
  - Search: hybrid text/vector search and result inspection

- Results (columns)
  - `#` — row number
  - `Chunk ID` — stored chunk id (click any cell to select the row)
  - `FTS` — SQLite FTS5 score (≈ normalized BM25)
  - `TV` — Tantivy default (QueryParser). A single-string query may behave like a strict phrase
  - `TV(AND)` — Lindera tokens combined with AND (all terms must match; BM25 scoring)
  - `TV(OR)` — Lindera tokens combined with OR (any term may match; BM25 scoring)
  - `VEC` — vector similarity from HNSW (≈ 0..1)
  - `Comb` — ordering score = 0.1·TV(AND) + 0.2·TV(OR) + 0.7·VEC
  - `Preview` — truncated text; click a row to see the full text below

- Housekeeping
  - Under Store/Index settings, use “Delete All (DB/HNSW/Tantivy)” (type `RESET` to enable) to reset data quickly
  - Japanese text rendering: CJK fallback font is auto-installed; to override, set `EMBEDDER_DEMO_FONT` to a font file path

- Notes
  - Tantivy is persisted on disk; opening a new (empty) Tantivy Dir will trigger a best-effort rebuild from SQLite
  - HNSW is persisted as a snapshot under the configured dir
  - The GUI displays multiple scores side-by-side to help interpret hybrid behaviour; only `Comb` is used for ordering by default


