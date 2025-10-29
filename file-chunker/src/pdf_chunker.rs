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

fn collect_text_and_boundaries(blocks: &[UnifiedBlock]) -> (String, Vec<Boundary>) {
    let mut text = String::new();
    let mut boundaries: Vec<Boundary> = Vec::new();

    let mut cursor = 0usize;
    for (i, b) in blocks.iter().enumerate() {
        // Normalize newlines
        let t = b.text.replace('\r', "");
        text.push_str(&t);
        cursor += t.len();
        // Prefer block boundaries
        if i + 1 < blocks.len() {
            boundaries.push(Boundary { idx: cursor, _kind: BoundaryKind::BlockEnd, base_score: 1.0 });
            // Add a single newline between blocks to avoid accidental merges
            text.push('\n');
            cursor += 1;
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

    (text, boundaries)
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
pub fn chunk_pdf_blocks_to_text(blocks: &[UnifiedBlock], params: &PdfChunkParams) -> Vec<String> {
    let (text, boundaries) = collect_text_and_boundaries(blocks);
    if text.trim().is_empty() { return vec![String::new()]; }
    // Precompute scored boundaries (sorted by idx)
    let mut scored: Vec<(usize, f32)> = boundaries
        .iter()
        .map(|b| (b.idx, penalize_after_short_line(&text, b)))
        .collect();
    scored.sort_by_key(|(i, _)| *i);

    let mut out: Vec<String> = Vec::new();
    let mut start = 0usize;
    let total = text.len();
    while start < total {
        let min = (start + params.min_chars).min(total);
        let max = (start + params.max_chars).min(total);
        let cap = (start + params.cap_chars).min(total);

        // If the remainder is small enough, flush and break
        if total - start <= params.cap_chars.max(1) {
            let seg = text[start..total].trim();
            if !seg.is_empty() { out.push(seg.to_string()); }
            break;
        }

        // Prefer boundary within [min..cap]
        if let Some(cut) = pick_boundary_in_range(&scored, min, cap, max) {
            if cut > start { let seg = text[start..cut].trim(); if !seg.is_empty() { out.push(seg.to_string()); } start = cut; continue; }
        }

        // Fallback: boundary just after cap, else last boundary
        let mut fallback_cut: Option<usize> = None;
        for (idx, _) in &scored { if *idx > cap { fallback_cut = Some(*idx); break; } }
        if fallback_cut.is_none() { if let Some((idx, _)) = scored.last() { fallback_cut = Some(*idx); } }
        let cut = fallback_cut.unwrap_or(total);
        if cut <= start || cut > total {
            let seg = text[start..total].trim();
            if !seg.is_empty() { out.push(seg.to_string()); }
            break;
        }
        let seg = text[start..cut].trim();
        if !seg.is_empty() { out.push(seg.to_string()); }
        start = cut;
    }

    if out.is_empty() { out.push(String::new()); }
    out
}

/// High-level: read PDF -> chunk -> return FileRecord and ChunkRecords
pub fn chunk_pdf_file_with_file_record(path: &str, params: &PdfChunkParams) -> (FileRecord, Vec<ChunkRecord>) {
    let blocks = read_pdf_to_blocks(path);
    let texts = chunk_pdf_blocks_to_text(&blocks, params);

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
        chunk_count: Some(texts.len() as u32),
        total_tokens: None,
        meta: BTreeMap::new(),
        extra: BTreeMap::new(),
    };

    let chunks: Vec<ChunkRecord> = texts
        .into_iter()
        .enumerate()
        .map(|(i, text)| ChunkRecord {
            schema_version: SCHEMA_MAJOR,
            doc_id: DocumentId(path.to_string()),
            chunk_id: ChunkId(format!("{}#{}", path, i)),
            source_uri: path.to_string(),
            source_mime: "application/pdf".into(),
            extracted_at: String::new(),
            text,
            section_path: None,
            meta: BTreeMap::new(),
            extra: BTreeMap::new(),
        })
        .collect();

    (file, chunks)
}
