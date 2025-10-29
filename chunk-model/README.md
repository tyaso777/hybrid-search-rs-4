## Chunk Model

[← Back to workspace README](../README.md)

Thin, shared schema crate that defines the data contract for content chunks produced by `file-chunker` and consumed by `chunking-store` and other tools.

## What it provides

- Core types: `DocumentId`, `ChunkId`, `SectionPath`, `ChunkRecord`, `FileRecord`
- Additional types: `BlockKind`, `LinkRef`
- Forward-compatible metadata bag via `#[serde(flatten)] extra`
- Minimal dependencies: `serde`, `serde_json`
- Soft validator: `ChunkRecord::validate_soft()`

## ChunkRecord (overview)

- `schema_version: u16` — major schema version for tolerant readers
- `doc_id: DocumentId` — opaque source document identifier
- `chunk_id: ChunkId` — opaque chunk identifier
- `source_uri: String` — origin (e.g., file path, URL)
- `source_mime: String` — MIME-like type (e.g., `application/pdf`)
- `extracted_at: String` — ISO 8601 (UTC), producer may leave empty
- `text: String` — searchable text body
- `section_path: Option<SectionPath>` — logical path within the document (optional)
- `meta: BTreeMap<String, String>` — lightweight key/value metadata
- `extra: BTreeMap<String, serde_json::Value>` — forward-compatible extensions (flattened)

## FileRecord (overview)

- `schema_version: u16`
- `doc_id: DocumentId`, `doc_revision: Option<u32)`
- `source_uri`, `source_mime`
- Optional file facts: `file_size_bytes`, `content_sha256`, `page_count`
- Timestamps (ISO 8601 UTC): `extracted_at`, `created_at_meta`, `updated_at_meta`
- Doc-level labels: `title_guess`, `author_guess`, `dominant_lang`, `tags`
- Ingestion info: `ingest_tool`, `ingest_tool_version`, `reader_backend`, `ocr_used`, `ocr_langs`
- Aggregates: `chunk_count`, `total_tokens`
- `meta`, `extra`

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
    section_path: Some(vec!["Ⅰ 概要".into()]),
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
  "section_path": null,
  "meta": {},
  "extra": {"layout.page": 1}
}
```

## Versioning and evolution

- Additive-first: introduce new fields under `extra` first; promote to core once stable.
- Compatibility: unknown fields are ignored thanks to `serde(flatten)`; older readers should parse safely.
- Breaking changes: bump `SCHEMA_MAJOR` and coordinate updates across stores/indexers/tools.

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

## Additional types

- `BlockKind` — suggested taxonomy for content kinds produced by upstream readers/normalizers.
  - Variants: `Heading`, `Paragraph`, `ListItem`, `Table`, `Figure`, `Caption`, `Code`, `Equation`, `Footnote`, `HeaderFooter`.
  - Guidance: if your chunking pipeline classifies blocks, you can carry the kind via a typed field in your own record or stash it under `extra` (e.g., `extra["layout.kind"] = "Heading"`).

- `LinkRef` — minimal inline link representation: `{ start, end, uri }` measured in UTF‑8 char indices.
  - Use when you want to preserve embedded hyperlinks for rendering or downstream processing. Store directly or under `extra` (e.g., `extra["links"] = [...]`).
