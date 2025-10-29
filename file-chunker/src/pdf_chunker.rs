use crate::reader_pdf::{read_pdf_to_blocks, default_backend, PdfBackend};
use crate::unified_blocks::UnifiedBlock;
use chunk_model::{ChunkRecord, DocumentId, ChunkId, FileRecord, SCHEMA_MAJOR};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub struct PdfChunkParams {
    /// Prefer chunk lengths >= this many characters
    pub min_chars: usize,
    /// Prefer chunk lengths around this many characters
    pub max_chars: usize,
    /// Hard cap: do not exceed this many characters per chunk when possible
    pub cap_chars: usize,
}

impl Default for PdfChunkParams {
    fn default() -> Self { Self { min_chars: 400, max_chars: 600, cap_chars: 800 } }
}

/// Produce segments from UnifiedBlocks using the generic text_segmenter.
fn chunk_pdf_blocks_to_segments(
    blocks: &[UnifiedBlock],
    params: &PdfChunkParams,
) -> Vec<(String, Option<u32>, Option<u32>)> {
    let tparams = crate::text_segmenter::TextChunkParams {
        min_chars: params.min_chars,
        max_chars: params.max_chars,
        cap_chars: params.cap_chars,
        penalize_short_line: true,
        penalize_page_boundary_no_newline: true,
    };
    crate::text_segmenter::chunk_blocks_to_segments(blocks, &tparams)
}

pub fn chunk_pdf_blocks_to_text(blocks: &[UnifiedBlock], params: &PdfChunkParams) -> Vec<String> {
    chunk_pdf_blocks_to_segments(blocks, params)
        .into_iter()
        .map(|(t, _, _)| t)
        .collect()
}

/// High-level: read PDF -> chunk -> return FileRecord and ChunkRecords
pub fn chunk_pdf_file_with_file_record(path: &str, params: &PdfChunkParams) -> (FileRecord, Vec<ChunkRecord>) {
    let blocks = read_pdf_to_blocks(path);
    let segs = chunk_pdf_blocks_to_segments(&blocks, params);

    let backend = match default_backend() {
        PdfBackend::Pdfium => "pdfium",
        PdfBackend::PureRust => "pure-pdf",
        PdfBackend::Stub => "stub.pdf",
    };

    let file = FileRecord {
        schema_version: SCHEMA_MAJOR,
        doc_id: DocumentId(path.to_string()),
        doc_revision: Some(1),
        source_uri: path.to_string(),
        source_mime: "application/pdf".into(),
        file_size_bytes: None,
        content_sha256: None,
        page_count: None,
        extracted_at: String::new(),
        created_at_meta: None,
        updated_at_meta: None,
        title_guess: None,
        author_guess: None,
        dominant_lang: None,
        tags: Vec::new(),
        ingest_tool: Some("pdf-chunker".into()),
        ingest_tool_version: Some(env!("CARGO_PKG_VERSION").into()),
        reader_backend: Some(backend.into()),
        ocr_used: None,
        ocr_langs: Vec::new(),
        chunk_count: Some(segs.len() as u32),
        total_tokens: None,
        meta: BTreeMap::new(),
        extra: BTreeMap::new(),
    };

    let chunks: Vec<ChunkRecord> = segs
        .into_iter()
        .enumerate()
        .map(|(i, (text, pstart, pend))| ChunkRecord {
            schema_version: SCHEMA_MAJOR,
            doc_id: DocumentId(path.to_string()),
            chunk_id: ChunkId(format!("{}#{}", path, i)),
            source_uri: path.to_string(),
            source_mime: "application/pdf".into(),
            extracted_at: String::new(),
            page_start: pstart,
            page_end: pend,
            text,
            section_path: None,
            meta: BTreeMap::new(),
            extra: BTreeMap::new(),
        })
        .collect();

    (file, chunks)
}

