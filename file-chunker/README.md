## File Chunker

[← Back to workspace README](../README.md)

Splits input files into logical text chunks for downstream indexing and retrieval.

### What It Does (current)
- Infers content type by extension (stub)
- Reads source into unified blocks (PDF/DOCX stubs)
- Applies JP chunking rules (stub)
- Emits `chunk_model::ChunkRecord` per chunk

### Status
- Scaffolding only. Real parsers, robust rules, and metadata are to be added.

### Output Shape
- See [chunk-model/README.md](../chunk-model/README.md) for the schema (`ChunkRecord`, `DocumentId`, `ChunkId`, `meta`, `extra`).

### Example (library)
```rust
use file_chunker::chunk_file;

let chunks = chunk_file("./docs/sample.pdf");
for c in chunks { println!("{}: {}", (c.chunk_id).0, c.text.len()); }
```
