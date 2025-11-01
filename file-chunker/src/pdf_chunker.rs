use crate::reader_pdf::{read_pdf_to_blocks, default_backend, PdfBackend};
use crate::unified_blocks::UnifiedBlock;
use chunk_model::{ChunkRecord, DocumentId, ChunkId, FileRecord, SCHEMA_MAJOR};
use std::collections::BTreeMap;
use chrono::{DateTime, Utc};
use sha2::Digest;
use std::fs::File as FsFile;
use std::io::{BufReader, Read};

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

/// Variant: produce segments using a provided TextChunkParams directly.
pub fn chunk_pdf_blocks_to_segments_with_text_params(
    blocks: &[UnifiedBlock],
    tparams: &crate::text_segmenter::TextChunkParams,
)
    -> Vec<(String, Option<u32>, Option<u32>)>
{
    crate::text_segmenter::chunk_blocks_to_segments(blocks, tparams)
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

    // Derive page count from segments' page_end
    let page_count = segs.iter().filter_map(|(_, _ps, pe)| *pe).max();

    let mut file = FileRecord {
        schema_version: SCHEMA_MAJOR,
        doc_id: DocumentId(path.to_string()),
        doc_revision: Some(1),
        source_uri: path.to_string(),
        source_mime: "application/pdf".into(),
        file_size_bytes: None,
        content_sha256: None,
        page_count,
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

    // Basic FS metadata + SHA256
    enrich_file_record_basic(&mut file, path);

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

// Reuse the same helpers as lib.rs (declare here to satisfy the compiler without extra pub exports)
fn enrich_file_record_basic(file: &mut chunk_model::FileRecord, path: &str) {
    if let Ok(md) = std::fs::metadata(path) {
        file.file_size_bytes = Some(md.len());
        if let Ok(ct) = md.created() { file.created_at_meta = Some(system_time_to_rfc3339(ct)); }
        if let Ok(mt) = md.modified() { file.updated_at_meta = Some(system_time_to_rfc3339(mt)); }
    }
    if let Some(hex) = compute_sha256_hex(path) { file.content_sha256 = Some(hex); }
}

fn compute_sha256_hex(path: &str) -> Option<String> {
    let f = FsFile::open(path).ok()?;
    let mut reader = BufReader::new(f);
    let mut hasher = sha2::Sha256::new();
    let mut buf = [0u8; 32 * 1024];
    loop {
        let n = reader.read(&mut buf).ok()?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Some(hex::encode(digest))
}

fn system_time_to_rfc3339(t: std::time::SystemTime) -> String {
    let dt: DateTime<Utc> = t.into();
    dt.to_rfc3339()
}
