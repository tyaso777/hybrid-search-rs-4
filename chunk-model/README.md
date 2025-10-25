# chunk-model

[← Back to workspace README](../README.md)

Thin, shared schema crate that defines the data contract for content chunks produced by `file-chunker` and consumed by `chunking-store` and other tools.

## What it provides

- Core types: `DocumentId`, `ChunkId`, `SectionPath`, `ChunkRecord`
- Forward-compatible metadata bag via `#[serde(flatten)] extra`
- Minimal dependencies: `serde`, `serde_json`
- Soft validator: `ChunkRecord::validate_soft()`

## Data model (overview)

- `schema_version: u16` — major schema version for tolerant readers
- `doc_id: DocumentId` — opaque source document identifier
- `chunk_id: ChunkId` — opaque chunk identifier
- `source_uri: String` — origin (e.g., file path, URL)
- `source_mime: String` — MIME-like type (e.g., `application/pdf`)
- `extracted_at: String` — ISO 8601 (UTC), producer may leave empty
- `text: String` — searchable text body
- `section_path: Vec<String>` — logical path within the document (optional)
- `meta: BTreeMap<String, String>` — lightweight key/value metadata
- `extra: BTreeMap<String, serde_json::Value>` — forward-compatible extensions (flattened)

## Example

```rust
use chunk_model::{ChunkRecord, DocumentId, ChunkId, SCHEMA_MAJOR};
use std::collections::BTreeMap;

let record = ChunkRecord {
    schema_version: SCHEMA_MAJOR,
    doc_id: DocumentId("doc-001".into()),
    chunk_id: ChunkId("doc-001#0".into()),
    source_uri: "./docs/sample.pdf".into(),
    source_mime: "application/pdf".into(),
    extracted_at: "".into(),
    text: "...chunk text...".into(),
    section_path: vec!["Ⅰ 概要".into()],
    meta: BTreeMap::new(),
    extra: BTreeMap::new(),
};
record.validate_soft().unwrap();
```

### JSON line (NDJSON) example

```json
{
  "schema_version": 1,
  "doc_id": "doc-001",
  "chunk_id": "doc-001#0",
  "source_uri": "./docs/sample.pdf",
  "source_mime": "application/pdf",
  "extracted_at": "",
  "text": "...chunk text...",
  "section_path": ["Ⅰ 概要"],
  "meta": {},
  "extra": {"layout.page": 1}
}
```

## Versioning and evolution

- Additive-first: 新しい情報はまず `extra` に追加し、十分に安定してからコア昇格
- 互換性: 未知フィールドは読み飛ばし（`serde(flatten)`）
- 破壊的変更: `SCHEMA_MAJOR` をインクリメントして連携側で対応

## How to depend (workspace)

```toml
[dependencies]
chunk-model = { path = "../chunk-model" }
```

## Optional features

Feature flags are reserved for small helpers to keep the base crate lean:

- `time` — helpers for ISO 8601 timestamps (planned)
- `hash` — deterministic IDs (e.g., blake3) (planned)
- `schema` — JSON Schema export (planned)
