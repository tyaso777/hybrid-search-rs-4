## File Chunker

[← Back to workspace README](../README.md)

Splits input files into logical text chunks for downstream indexing and retrieval.

### What It Does (current)
- Infers content type by extension
- Reads source into unified blocks:
  - PDF (pdfium or pure-pdf backends)
  - DOCX (XML parse)
  - TXT (UTF‑8 / optional encodings)
  - Excel: XLSX/XLS/ODS (via calamine)
- Segments text with a unified segmenter and emits `chunk_model::ChunkRecord` per chunk

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

### PDFium Backend (Windows x64)

This crate can read PDFs via two backends:

- `pdfium` (recommended): fast and robust using the `pdfium-render` crate
- `pure-pdf` (fallback): pure Rust parser with limited fidelity

In this workspace, the service enables both features for `file-chunker`, so PDFium will be used when its DLL is found.

PDFium is located at runtime in the following order:

1) Environment variables
   - `PDFIUM_DLL_PATH`: full path to `pdfium.dll` or a directory containing it
   - `PDFIUM_DIR`: directory that contains `pdfium.dll`
2) Bundled locations under this crate
   - `file-chunker/bin/pdfium-win-x64/bin/pdfium.dll`
   - `file-chunker/bin/pdfium-win-x64/pdfium.dll`
   - `file-chunker/bin/pdfium.dll`

Recommended setup on Windows (PowerShell):

1) Download PDFium (Windows x64) prebuilt binary
   - Get it from the community builds (e.g. pdfium-binaries):
     https://github.com/bblanchon/pdfium-binaries/releases
   - Download the archive for Windows x64 (MSVC) and extract it.

2) Place the DLL into this repository
   - Copy `pdfium.dll` to:
     `file-chunker/bin/pdfium-win-x64/bin/pdfium.dll`
   - Optional: headers may be kept in
     `file-chunker/bin/pdfium-win-x64/include` (not required for runtime).

3) Alternatively, use environment variables
   - One‑shot in the current shell:
     `$env:PDFIUM_DLL_PATH = "D:\Users\AtsushiSuzuki\docs\rust\hybrid-search-rs-4\file-chunker\bin\pdfium-win-x64\bin\pdfium.dll"`
   - Persist for the user:
     `setx PDFIUM_DLL_PATH "D:\Users\AtsushiSuzuki\docs\rust\hybrid-search-rs-4\file-chunker\bin\pdfium-win-x64\bin\pdfium.dll"`

4) Verify
   - Run the PDF block viewer: `cargo run -p pdf-block-viewer`
   - Or ingest a PDF via the service/GUI and ensure the backend shown is `pdfium`.

Notes
- Use the 64‑bit DLL for the default `x86_64-pc-windows-msvc` toolchain.
- If no DLL is found, the code falls back to the pure‑Rust backend.

#### Licenses / Notices (distribution)

When you redistribute an app that uses PDFium, include the vendor licenses and notices alongside your binary.

- Place the files next to your EXE (or in your app bundle) together with `pdfium.dll`.
- Typical files from the PDFium binary archive:
  - `LICENSE`
  - `ThirdPartyNotices.txt` (or similar)
- If you copy `pdfium.dll` under this repo (for development), keep the notices here as well:
  - `file-chunker/bin/pdfium-win-x64/LICENSE`
  - Optionally: `file-chunker/bin/pdfium-win-x64/ThirdPartyNotices.txt`

See also: `docs/compliance.md` for generating third‑party license summaries/texts for this workspace.
