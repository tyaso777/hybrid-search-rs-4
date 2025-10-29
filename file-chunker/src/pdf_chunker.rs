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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoundaryKind { BlockEnd, DoubleNewline, SingleNewline, SentenceEnd }

#[derive(Debug, Clone)]
struct Boundary { idx: usize, _kind: BoundaryKind, base_score: f32 }

#[derive(Debug, Clone, Copy)]
struct BlockSpan { start: usize, end: usize, page_start: Option<u32>, page_end: Option<u32> }

fn collect_text_and_boundaries(blocks: &[UnifiedBlock]) -> (String, Vec<Boundary>, Vec<BlockSpan>) {
    let mut text = String::new();
    let mut boundaries: Vec<Boundary> = Vec::new();
    let mut spans: Vec<BlockSpan> = Vec::new();

    let mut cursor = 0usize;
    for (i, b) in blocks.iter().enumerate() {
        // Normalize newlines
        let t = b.text.replace('\r', "");
        let start_idx = cursor;
        text.push_str(&t);
        cursor += t.len();
        spans.push(BlockSpan { start: start_idx, end: cursor, page_start: b.page_start, page_end: b.page_end });
        // Prefer block boundaries
        if i + 1 < blocks.len() {
            // Prefer block boundaries, but do not inject artificial newlines between blocks.
            boundaries.push(Boundary { idx: cursor, _kind: BoundaryKind::BlockEnd, base_score: 1.0 });
        }
    }

    // Scan for newline and sentence boundaries
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'\n' => {
                // Double newline?
                if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                    boundaries.push(Boundary { idx: i + 2, _kind: BoundaryKind::DoubleNewline, base_score: 0.95 });
                    i += 2;
                    continue;
                } else {
                    boundaries.push(Boundary { idx: i + 1, _kind: BoundaryKind::SingleNewline, base_score: 0.8 });
                }
            }
            _ => {}
        }
        i += 1;
    }

    // Sentence ends (basic ASCII and common JP full stops)
    for (idx, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?' | '。' | '！' | '？' | '．') {
            boundaries.push(Boundary { idx: idx + ch.len_utf8(), _kind: BoundaryKind::SentenceEnd, base_score: 0.6 });
        }
    }

    // Sort and dedup boundary indices, keep best base_score per idx
    boundaries.sort_by_key(|b| b.idx);
    boundaries.dedup_by(|a, b| {
        if a.idx == b.idx {
            // Keep the one with higher base score
            if a.base_score < b.base_score { a.base_score = b.base_score; }
            true
        } else { false }
    });

    (text, boundaries, spans)
}

fn penalize_after_short_line(text: &str, b: &Boundary) -> f32 {
    // Avoid splitting immediately after a very short line (heuristic)
    let mut j = if b.idx > 0 { b.idx - 1 } else { 0 };
    // Find start of current line
    while j > 0 && text.as_bytes()[j] != b'\n' { j -= 1; }
    let line_start = if text.as_bytes()[j] == b'\n' { j + 1 } else { j };
    let line_len = b.idx.saturating_sub(line_start);
    let penalty = if line_len < 10 { 0.35 } else { 0.0 };
    b.base_score - penalty
}

// Pick best boundary between [lo..=hi], preferring near `prefer` and higher score.
fn pick_boundary_in_range(scored: &[(usize, f32)], lo: usize, hi: usize, prefer: usize) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (idx, score) in scored {
        if *idx < lo || *idx > hi { continue; }
        let dist = if *idx > prefer { *idx - prefer } else { prefer - *idx } as f32;
        let eff = *score - dist / ((hi.saturating_sub(lo)) as f32 + 1.0);
        if let Some((_, b)) = best { if eff > b { best = Some((*idx, eff)); } } else { best = Some((*idx, eff)); }
    }
    best.map(|(i, _)| i)
}

// removed

/// Produce ChunkRecord texts from UnifiedBlocks with heuristics.
fn chunk_pdf_blocks_to_segments(blocks: &[UnifiedBlock], params: &PdfChunkParams) -> Vec<(String, Option<u32>, Option<u32>)> {
    let tparams = crate::text_segmenter::TextChunkParams {
        min_chars: params.min_chars,
        max_chars: params.max_chars,
        cap_chars: params.cap_chars,
        penalize_short_line: true,
        penalize_page_boundary_no_newline: true,
    };
    crate::text_segmenter::chunk_blocks_to_segments(blocks, &tparams)
}

fn extra_penalty_page_boundary_no_newline(idx: usize, text: &str, spans: &[BlockSpan]) -> f32 {
    // Find if `idx` equals a block end and the next block starts on a different page,
    // and the preceding character is not a newline. If so, penalize to encourage merging.
    for w in spans.windows(2) {
        let a = &w[0];
        let b = &w[1];
        if a.end == idx {
            let page_transition = match (a.page_end, b.page_start) {
                (Some(pe), Some(ps)) => pe != ps,
                _ => false,
            };
            let has_newline_before = idx > 0 && text.as_bytes()[idx.saturating_sub(1)] == b'\n';
            if page_transition && !has_newline_before { return 0.4; }
        }
    }
    0.0
}

fn page_range_for_segment(start: usize, end: usize, spans: &[BlockSpan]) -> (Option<u32>, Option<u32>) {
    let mut min_p: Option<u32> = None;
    let mut max_p: Option<u32> = None;
    for s in spans {
        if s.end <= start || s.start >= end { continue; }
        if let Some(ps) = s.page_start { min_p = Some(match min_p { Some(v) => v.min(ps), None => ps }); }
        if let Some(pe) = s.page_end { max_p = Some(match max_p { Some(v) => v.max(pe), None => pe }); }
    }
    (min_p, max_p)
}

pub fn chunk_pdf_blocks_to_text(blocks: &[UnifiedBlock], params: &PdfChunkParams) -> Vec<String> {
    chunk_pdf_blocks_to_segments(blocks, params).into_iter().map(|(t, _, _)| t).collect()
}

/// High-level: read PDF -> chunk -> return FileRecord and ChunkRecords
pub fn chunk_pdf_file_with_file_record(path: &str, params: &PdfChunkParams) -> (FileRecord, Vec<ChunkRecord>) {
    let blocks = read_pdf_to_blocks(path);
    let segs = chunk_pdf_blocks_to_segments(&blocks, params);

    let backend = match default_backend() { PdfBackend::Pdfium => "pdfium", PdfBackend::PureRust => "pure-pdf", PdfBackend::Stub => "stub.pdf" };

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
