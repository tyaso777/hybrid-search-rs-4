pub mod reader_pdf;
pub mod reader_docx;
pub mod reader_txt;
pub mod reader_excel;
pub mod reader_pptx;
pub mod unified_blocks;
pub mod chunker_rules_jp;
pub mod text_segmenter;
#[cfg(feature = "pdfium")] pub mod reader_pdf_pdfium;
#[cfg(feature = "pure-pdf")] pub mod reader_pdf_pure;
pub mod pdf_chunker;

use chunk_model::{ChunkId, ChunkRecord, DocumentId, FileRecord};
use std::fs::File;
use std::io::{BufReader, Read};
use chrono::{DateTime, Utc};
use sha2::Digest;
use std::collections::BTreeMap;
use unified_blocks::{UnifiedBlock, BlockKind};
use std::path::Path;

/// Result bundle including file-level metadata and chunk list.
#[derive(Debug, Clone)]
pub struct ChunkOutput {
    pub file: FileRecord,
    pub chunks: Vec<ChunkRecord>,
}

/// Unified options for chunking behavior.
#[derive(Debug, Clone)]
pub struct ChunkOptions {
    /// Encoding hint for text-like files (e.g., "shift_jis"). None means auto/UTF-8.
    pub encoding: Option<String>,
    /// Optional text segmentation parameters override.
    pub params: Option<text_segmenter::TextChunkParams>,
}

impl Default for ChunkOptions {
    fn default() -> Self { Self { encoding: None, params: None } }
}

/// Unified entry to chunk a file with optional encoding and segmentation params.
pub fn chunk_file_with_file_record_with_options(path: &str, opts: &ChunkOptions) -> ChunkOutput {
    let lower = path.to_lowercase();

    // PDF
    if lower.ends_with(".pdf") {
        if let Some(p) = opts.params {
            let blocks: Vec<UnifiedBlock> = reader_pdf::read_pdf_to_blocks(path);
            let segs = pdf_chunker::chunk_pdf_blocks_to_segments_with_text_params(&blocks, &p);
            let page_count = segs.iter().filter_map(|(_, _ps, pe)| *pe).max();
            let chunks: Vec<ChunkRecord> = segs
                .into_iter()
                .enumerate()
                .map(|(i, (text, ps, pe))| ChunkRecord {
                    schema_version: chunk_model::SCHEMA_MAJOR,
                    doc_id: DocumentId(path.to_string()),
                    chunk_id: ChunkId(format!("{}#{}", path, i)),
                    source_uri: path.to_string(),
                    source_mime: "application/pdf".into(),
                    extracted_at: String::new(),
                    page_start: ps,
                    page_end: pe,
                    text,
                    section_path: None,
                    meta: BTreeMap::new(),
                    extra: BTreeMap::new(),
                })
                .collect();
            let mut file = FileRecord {
                schema_version: chunk_model::SCHEMA_MAJOR,
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
                ingest_tool: Some("file-chunker".into()),
                ingest_tool_version: Some(env!("CARGO_PKG_VERSION").into()),
                reader_backend: Some("pdf".into()),
                ocr_used: None,
                ocr_langs: Vec::new(),
                chunk_count: Some(chunks.len() as u32),
                total_tokens: None,
                meta: BTreeMap::new(),
                extra: BTreeMap::new(),
            };
            enrich_file_record_basic(&mut file, path);
            return ChunkOutput { file, chunks };
        } else {
            let params = pdf_chunker::PdfChunkParams::default();
            let (file, chunks) = pdf_chunker::chunk_pdf_file_with_file_record(path, &params);
            return ChunkOutput { file, chunks };
        }
    }

    // DOCX (derive cut levels dynamically)
    if lower.ends_with(".docx") {
        let blocks: Vec<UnifiedBlock> = reader_docx::read_docx_to_blocks(path);
        let params = opts.params.unwrap_or_else(text_segmenter::TextChunkParams::default);
        let levels = derive_docx_cut_levels(&blocks);
        let segs = if levels.is_empty() {
            text_segmenter::chunk_blocks_to_segments(&blocks, &params)
        } else {
            let pair = if levels.len() >= 2 { Some((levels[0], levels[1])) } else { None };
            chunk_blocks_grouped_by_levels(&blocks, &params, &levels, pair)
        };
        let page_count = segs.iter().filter_map(|(_, _ps, pe)| *pe).max();
        let chunks: Vec<ChunkRecord> = segs
            .into_iter()
            .enumerate()
            .map(|(i, (text, ps, pe))| ChunkRecord {
                schema_version: chunk_model::SCHEMA_MAJOR,
                doc_id: DocumentId(path.to_string()),
                chunk_id: ChunkId(format!("{}#{}", path, i)),
                source_uri: path.to_string(),
                source_mime: "application/vnd.openxmlformats-officedocument.wordprocessingml.document".into(),
                extracted_at: String::new(),
                page_start: ps,
                page_end: pe,
                text,
                section_path: None,
                meta: BTreeMap::new(),
                extra: BTreeMap::new(),
            })
            .collect();
        let mut file = FileRecord {
            schema_version: chunk_model::SCHEMA_MAJOR,
            doc_id: DocumentId(path.to_string()),
            doc_revision: Some(1),
            source_uri: path.to_string(),
            source_mime: "application/vnd.openxmlformats-officedocument.wordprocessingml.document".into(),
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
            ingest_tool: Some("file-chunker".into()),
            ingest_tool_version: Some(env!("CARGO_PKG_VERSION").into()),
            reader_backend: Some("docx".into()),
            ocr_used: None,
            ocr_langs: Vec::new(),
            chunk_count: Some(chunks.len() as u32),
            total_tokens: None,
            meta: BTreeMap::new(),
            extra: BTreeMap::new(),
        };
        enrich_file_record_basic(&mut file, path);
        return ChunkOutput { file, chunks };
    }

    // PPTX (slides as H1 boundaries; tables honored)
    if lower.ends_with(".pptx") {
        let blocks: Vec<UnifiedBlock> = reader_pptx::read_pptx_to_blocks(path);
        let params = opts.params.unwrap_or_else(text_segmenter::TextChunkParams::default);
        let segs = chunk_blocks_grouped_by_h1(&blocks, &params);
        let page_count = segs.iter().filter_map(|(_, _ps, pe)| *pe).max();
        let chunks: Vec<ChunkRecord> = segs
            .into_iter()
            .enumerate()
            .map(|(i, (text, ps, pe))| ChunkRecord {
                schema_version: chunk_model::SCHEMA_MAJOR,
                doc_id: DocumentId(path.to_string()),
                chunk_id: ChunkId(format!("{}#{}", path, i)),
                source_uri: path.to_string(),
                source_mime: "application/vnd.openxmlformats-officedocument.presentationml.presentation".into(),
                extracted_at: String::new(),
                page_start: ps,
                page_end: pe,
                text,
                section_path: None,
                meta: BTreeMap::new(),
                extra: BTreeMap::new(),
            })
            .collect();

        let mut file = FileRecord {
            schema_version: chunk_model::SCHEMA_MAJOR,
            doc_id: DocumentId(path.to_string()),
            doc_revision: Some(1),
            source_uri: path.to_string(),
            source_mime: "application/vnd.openxmlformats-officedocument.presentationml.presentation".into(),
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
            ingest_tool: Some("file-chunker".into()),
            ingest_tool_version: Some(env!("CARGO_PKG_VERSION").into()),
            reader_backend: Some("pptx".into()),
            ocr_used: None,
            ocr_langs: Vec::new(),
            chunk_count: Some(chunks.len() as u32),
            total_tokens: None,
            meta: BTreeMap::new(),
            extra: BTreeMap::new(),
        };
        enrich_file_record_basic(&mut file, path);
        return ChunkOutput { file, chunks };
    }
    // Excel
    if lower.ends_with(".xlsx") || lower.ends_with(".xls") || lower.ends_with(".ods") {
        let blocks: Vec<UnifiedBlock> = reader_excel::read_excel_to_blocks(path);
        let params = opts.params.unwrap_or_else(text_segmenter::TextChunkParams::default);
        let segs = chunk_blocks_grouped_by_h1(&blocks, &params);
        let page_count = segs.iter().filter_map(|(_, _ps, pe)| *pe).max();

        let src_mime = if lower.ends_with(".xlsx") {
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        } else if lower.ends_with(".xls") {
            "application/vnd.ms-excel"
        } else { "application/vnd.oasis.opendocument.spreadsheet" };

        let chunks: Vec<ChunkRecord> = segs
            .into_iter()
            .enumerate()
            .map(|(i, (text, ps, pe))| ChunkRecord {
                schema_version: chunk_model::SCHEMA_MAJOR,
                doc_id: DocumentId(path.to_string()),
                chunk_id: ChunkId(format!("{}#{}", path, i)),
                source_uri: path.to_string(),
                source_mime: src_mime.into(),
                extracted_at: String::new(),
                page_start: ps,
                page_end: pe,
                text,
                section_path: None,
                meta: BTreeMap::new(),
                extra: BTreeMap::new(),
            })
            .collect();

        let mut file = FileRecord {
            schema_version: chunk_model::SCHEMA_MAJOR,
            doc_id: DocumentId(path.to_string()),
            doc_revision: Some(1),
            source_uri: path.to_string(),
            source_mime: src_mime.into(),
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
            ingest_tool: Some("file-chunker".into()),
            ingest_tool_version: Some(env!("CARGO_PKG_VERSION").into()),
            reader_backend: Some("excel".into()),
            ocr_used: None,
            ocr_langs: Vec::new(),
            chunk_count: Some(chunks.len() as u32),
            total_tokens: None,
            meta: BTreeMap::new(),
            extra: BTreeMap::new(),
        };
        enrich_file_record_basic(&mut file, path);
        return ChunkOutput { file, chunks };
    }

    // Text-like
    if is_text_like(path) {
        let blocks: Vec<UnifiedBlock> = match &opts.encoding {
            Some(enc) => reader_txt::read_txt_to_blocks_with_encoding(path, Some(enc.as_str())),
            None => reader_txt::read_txt_to_blocks(path),
        };
        let params = opts.params.unwrap_or_else(text_segmenter::TextChunkParams::default);
        let segs = text_segmenter::chunk_blocks_to_segments(&blocks, &params);
        let chunks: Vec<ChunkRecord> = segs
            .into_iter()
            .enumerate()
            .map(|(i, (text, _ps, _pe))| ChunkRecord {
                schema_version: chunk_model::SCHEMA_MAJOR,
                doc_id: DocumentId(path.to_string()),
                chunk_id: ChunkId(format!("{}#{}", path, i)),
                source_uri: path.to_string(),
                source_mime: "text/plain".into(),
                extracted_at: String::new(),
                page_start: Some(1),
                page_end: Some(1),
                text,
                section_path: None,
                meta: BTreeMap::new(),
                extra: BTreeMap::new(),
            })
            .collect();
        let mut file = FileRecord {
            schema_version: chunk_model::SCHEMA_MAJOR,
            doc_id: DocumentId(path.to_string()),
            doc_revision: Some(1),
            source_uri: path.to_string(),
            source_mime: "text/plain".into(),
            file_size_bytes: None,
            content_sha256: None,
            page_count: Some(1),
            extracted_at: String::new(),
            created_at_meta: None,
            updated_at_meta: None,
            title_guess: None,
            author_guess: None,
            dominant_lang: None,
            tags: Vec::new(),
            ingest_tool: Some("file-chunker".into()),
            ingest_tool_version: Some(env!("CARGO_PKG_VERSION").into()),
            reader_backend: Some("txt".into()),
            ocr_used: None,
            ocr_langs: Vec::new(),
            chunk_count: Some(chunks.len() as u32),
            total_tokens: None,
            meta: BTreeMap::new(),
            extra: BTreeMap::new(),
        };
        enrich_file_record_basic(&mut file, path);
        return ChunkOutput { file, chunks };
    }

    // Fallback stub
    let blocks: Vec<UnifiedBlock> = vec![UnifiedBlock::new(BlockKind::Paragraph, "(stub) read file content here", 0, path, "stub.plain")];
    let texts = chunker_rules_jp::chunk_blocks_jp(&blocks);
    let chunks: Vec<ChunkRecord> = texts
        .into_iter()
        .enumerate()
        .map(|(i, text)| ChunkRecord {
            schema_version: chunk_model::SCHEMA_MAJOR,
            doc_id: DocumentId(path.to_string()),
            chunk_id: ChunkId(format!("{}#{}", path, i)),
            source_uri: path.to_string(),
            source_mime: "text/plain".into(),
            extracted_at: String::new(),
            page_start: None,
            page_end: None,
            text,
            section_path: None,
            meta: BTreeMap::new(),
            extra: BTreeMap::new(),
        })
        .collect();

    let mut file = FileRecord {
        schema_version: chunk_model::SCHEMA_MAJOR,
        doc_id: DocumentId(path.to_string()),
        doc_revision: Some(1),
        source_uri: path.to_string(),
        source_mime: "text/plain".into(),
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
        ingest_tool: Some("file-chunker".into()),
        ingest_tool_version: Some(env!("CARGO_PKG_VERSION").into()),
        reader_backend: Some("stub".into()),
        ocr_used: None,
        ocr_langs: Vec::new(),
        chunk_count: Some(chunks.len() as u32),
        total_tokens: None,
        meta: BTreeMap::new(),
        extra: BTreeMap::new(),
    };
    enrich_file_record_basic(&mut file, path);
    ChunkOutput { file, chunks }
}

/// High-level entry to chunk a file by path and return file/chunks.
/// This is a stubbed pipeline that returns a simple chunking for now.
pub fn chunk_file_with_file_record(path: &str) -> ChunkOutput {
    chunk_file_with_file_record_with_options(path, &ChunkOptions::default())
}

/// Variant with an explicit encoding hint for text-like files.
/// For non-text formats (PDF/DOCX), the behavior is identical to `chunk_file_with_file_record`.
pub fn chunk_file_with_file_record_with_encoding(path: &str, encoding: Option<&str>) -> ChunkOutput {
    let opts = ChunkOptions { encoding: encoding.map(|s| s.to_string()), params: None };
    chunk_file_with_file_record_with_options(path, &opts)
}

/// Unified entry with explicit TextChunkParams and optional encoding for text-like files.
pub fn chunk_file_with_file_record_with_params(
    path: &str,
    encoding: Option<&str>,
    params: &text_segmenter::TextChunkParams,
) -> ChunkOutput {
    let opts = ChunkOptions { encoding: encoding.map(|s| s.to_string()), params: Some(*params) };
    chunk_file_with_file_record_with_options(path, &opts)
}

/// Legacy helper returning only chunks for backward compatibility.
pub fn chunk_file(path: &str) -> Vec<ChunkRecord> {
    chunk_file_with_file_record(path).chunks
}

fn is_text_like(path: &str) -> bool {
    let lower = path.to_lowercase();
    // Common text-ish extensions
    let exts = [
        ".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".json", ".yaml", ".yml",
        ".ini", ".toml", ".cfg", ".conf", ".rst", ".tex", ".srt", ".properties",
    ];
    if exts.iter().any(|e| lower.ends_with(e)) {
        return true;
    }
    // As a fallback, if no extension or unknown, try a lightweight probe: small read and check for NUL bytes
    if Path::new(path).extension().is_none() {
        use std::io::Read;
        if let Ok(mut f) = std::fs::File::open(path) {
            let mut buf = [0u8; 2048];
            if let Ok(n) = f.read(&mut buf) {
                let slice = &buf[..n];
                return !slice.iter().any(|&b| b == 0);
            }
        }
    }
    false
}

/// Split blocks on top-level heading (Heading with level==1) and apply the generic text segmenter per group.
/// This enforces that no chunk crosses a top-level heading boundary. If no such headings exist, the
/// entire block list is treated as a single group.
fn chunk_blocks_grouped_by_h1(
    blocks: &[UnifiedBlock],
    params: &text_segmenter::TextChunkParams,
) -> Vec<(String, Option<u32>, Option<u32>)> {
    let mut out: Vec<(String, Option<u32>, Option<u32>)> = Vec::new();
    let mut cur: Vec<UnifiedBlock> = Vec::new();
    for b in blocks.iter() {
        let is_h1 = matches!(b.kind, BlockKind::Heading) && matches!(b.heading_level, Some(1));
        if is_h1 && !cur.is_empty() {
            let mut segs = text_segmenter::chunk_blocks_to_segments(&cur, params);
            out.append(&mut segs);
            cur.clear();
        }
        cur.push(b.clone());
    }
    if !cur.is_empty() {
        let mut segs = text_segmenter::chunk_blocks_to_segments(&cur, params);
        out.append(&mut segs);
    }
    out
}

/// Determine DOCX cut levels dynamically:
/// - Count heading occurrences per level (1..9)
/// - Ignore levels that appear only once
/// - Pick the first level (from 1 upward) with count >= 2 as "chapter"
/// - Pick the next such level as "section" (if any)
fn derive_docx_cut_levels(blocks: &[UnifiedBlock]) -> Vec<u8> {
    let mut counts: [u32; 10] = [0; 10];
    for b in blocks {
        if let (BlockKind::Heading, Some(lv)) = (b.kind.clone(), b.heading_level) {
            if (1..=9).contains(&lv) { counts[lv as usize] += 1; }
        }
    }
    let mut out: Vec<u8> = Vec::new();
    for lv in 1u8..=9u8 {
        if counts[lv as usize] >= 2 {
            out.push(lv);
            if out.len() >= 2 { break; }
        }
    }
    out
}

/// Split blocks when encountering a Heading whose level is in `cut_levels`.
/// When `skip_h1_h2_gap` is true, do not cut between a newly-started H1 and an immediately-following H2
/// (keeps H1 and its first H2 together at the start of the group).
fn chunk_blocks_grouped_by_levels(
    blocks: &[UnifiedBlock],
    params: &text_segmenter::TextChunkParams,
    cut_levels: &[u8],
    no_cut_pair: Option<(u8, u8)>,
) -> Vec<(String, Option<u32>, Option<u32>)> {
    let mut out: Vec<(String, Option<u32>, Option<u32>)> = Vec::new();
    let mut cur: Vec<UnifiedBlock> = Vec::new();
    for b in blocks.iter() {
        let is_heading = matches!(b.kind, BlockKind::Heading);
        let b_level = b.heading_level.unwrap_or(0);
        let is_boundary = is_heading && cut_levels.contains(&b_level);
        if is_boundary {
            let mut need_flush = !cur.is_empty();
            // Optional rule: if the boundary is "section" and the current group contains only
            // a single preceding "chapter" heading, do not cut between them.
            if need_flush {
                if let Some((chap_lv, sec_lv)) = no_cut_pair {
                    if b_level == sec_lv && cur.len() == 1 {
                        let last = &cur[0];
                        if matches!(last.kind, BlockKind::Heading) && last.heading_level == Some(chap_lv) {
                            need_flush = false;
                        }
                    }
                }
            }
            if need_flush {
                let mut segs = text_segmenter::chunk_blocks_to_segments(&cur, params);
                out.append(&mut segs);
                cur.clear();
            }
            cur.push(b.clone());
        } else {
            cur.push(b.clone());
        }
    }
    if !cur.is_empty() {
        let mut segs = text_segmenter::chunk_blocks_to_segments(&cur, params);
        out.append(&mut segs);
    }
    out
}

// --- Metadata enrichment helpers --------------------------------------------------------------

fn enrich_file_record_basic(file: &mut FileRecord, path: &str) {
    // File size and timestamps
    if let Ok(md) = std::fs::metadata(path) {
        file.file_size_bytes = Some(md.len());
        if let Ok(ct) = md.created() {
            file.created_at_meta = Some(system_time_to_rfc3339(ct));
        }
        if let Ok(mt) = md.modified() {
            file.updated_at_meta = Some(system_time_to_rfc3339(mt));
        }
    }
    // SHA-256 of file content
    if let Some(hex) = compute_sha256_hex(path) {
        file.content_sha256 = Some(hex);
    }
    // Windows: fallback author from file owner when not present
    #[cfg(target_os = "windows")]
    {
        if file.author_guess.is_none() {
            if let Some(owner) = windows_file_owner(path) { file.author_guess = Some(owner); }
        }
    }
}

fn compute_sha256_hex(path: &str) -> Option<String> {
    let f = File::open(path).ok()?;
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

#[cfg(target_os = "windows")]
fn windows_file_owner(path: &str) -> Option<String> {
    use std::iter;
    use std::os::windows::ffi::OsStrExt;
    use std::ptr::{null, null_mut};
    use windows::core::{PCWSTR, PWSTR};
    use windows::Win32::Foundation::{BOOL, PSID};
    use windows::Win32::Security::{GetFileSecurityW, GetSecurityDescriptorOwner, LookupAccountSidW, OWNER_SECURITY_INFORMATION, SID_NAME_USE, PSECURITY_DESCRIPTOR};

    let wide: Vec<u16> = std::ffi::OsStr::new(path).encode_wide().chain(iter::once(0)).collect();
    unsafe {
        let mut needed: u32 = 0;
        // First call to get needed buffer length
        let _ = GetFileSecurityW(PCWSTR(wide.as_ptr()), OWNER_SECURITY_INFORMATION.0, PSECURITY_DESCRIPTOR(null_mut()), 0, &mut needed);
        if needed == 0 { return None; }
        let mut buf: Vec<u8> = vec![0; needed as usize];
        let ok = GetFileSecurityW(PCWSTR(wide.as_ptr()), OWNER_SECURITY_INFORMATION.0, PSECURITY_DESCRIPTOR(buf.as_mut_ptr() as *mut _), needed, &mut needed);
        if ok.0 == 0 { return None; }
        let mut owner_sid: PSID = PSID(null_mut());
        let mut defaulted = BOOL(0);
        let sd = PSECURITY_DESCRIPTOR(buf.as_mut_ptr() as *mut _);
        if GetSecurityDescriptorOwner(sd, &mut owner_sid, &mut defaulted).is_err() || owner_sid.0.is_null() { return None; }

        // Query sizes
        let mut name_len: u32 = 0;
        let mut domain_len: u32 = 0;
        let mut use_: SID_NAME_USE = SID_NAME_USE(0);
        let _ = LookupAccountSidW(PCWSTR(null()), owner_sid, PWSTR(null_mut()), &mut name_len, PWSTR(null_mut()), &mut domain_len, &mut use_);
        if name_len == 0 { name_len = 256; }
        if domain_len == 0 { domain_len = 256; }
        let mut name_buf: Vec<u16> = vec![0u16; name_len as usize];
        let mut domain_buf: Vec<u16> = vec![0u16; domain_len as usize];
        let mut use2: SID_NAME_USE = SID_NAME_USE(0);
        let ok3 = LookupAccountSidW(
            PCWSTR(null()),
            owner_sid,
            PWSTR(name_buf.as_mut_ptr()),
            &mut name_len,
            PWSTR(domain_buf.as_mut_ptr()),
            &mut domain_len,
            &mut use2,
        );
        if ok3.is_err() { return None; }
        let name = String::from_utf16_lossy(&name_buf[..name_len as usize]);
        let domain = String::from_utf16_lossy(&domain_buf[..domain_len as usize]);
        if domain.is_empty() { Some(name) } else { Some(format!("{}\\{}", domain, name)) }
    }
}

