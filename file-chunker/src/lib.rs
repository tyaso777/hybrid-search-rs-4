pub mod reader_pdf;
pub mod reader_docx;
pub mod reader_txt;
pub mod reader_excel;
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

/// High-level entry to chunk a file by path and return file/chunks.
/// This is a stubbed pipeline that returns a simple chunking for now.
pub fn chunk_file_with_file_record(path: &str) -> ChunkOutput {
    let lower = path.to_lowercase();
    // PDF: delegate to dedicated chunker which already returns page ranges
    if lower.ends_with(".pdf") {
        let params = pdf_chunker::PdfChunkParams::default();
        let (file, chunks) = pdf_chunker::chunk_pdf_file_with_file_record(path, &params);
        return ChunkOutput { file, chunks };
    }

    // DOCX: read blocks, segment with text_segmenter (hard-cut at top-level headings)
    if lower.ends_with(".docx") {
        let blocks: Vec<UnifiedBlock> = reader_docx::read_docx_to_blocks(path);
        let params = text_segmenter::TextChunkParams::default();
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

    // Excel (.xlsx/.xls/.ods): read blocks and segment (hard-cut at sheet = top-level heading)
    if lower.ends_with(".xlsx") || lower.ends_with(".xls") || lower.ends_with(".ods") {
        let blocks: Vec<UnifiedBlock> = reader_excel::read_excel_to_blocks(path);
        let params = text_segmenter::TextChunkParams::default();
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

    // Text-like files: segment (hard-cut at top-level headings when present) and set page to 1
    if is_text_like(path) {
        let blocks: Vec<UnifiedBlock> = reader_txt::read_txt_to_blocks(path);
        let params = text_segmenter::TextChunkParams::default();
        let segs = chunk_blocks_grouped_by_h1(&blocks, &params);
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

/// Variant with an explicit encoding hint for text-like files.
/// For non-text formats (PDF/DOCX), the behavior is identical to `chunk_file_with_file_record`.
pub fn chunk_file_with_file_record_with_encoding(path: &str, encoding: Option<&str>) -> ChunkOutput {
    let lower = path.to_lowercase();
    // PDF
    if lower.ends_with(".pdf") {
        let params = pdf_chunker::PdfChunkParams::default();
        let (file, chunks) = pdf_chunker::chunk_pdf_file_with_file_record(path, &params);
        return ChunkOutput { file, chunks };
    }

    // DOCX (hard-cut at top-level headings)
    if lower.ends_with(".docx") {
        let blocks: Vec<UnifiedBlock> = reader_docx::read_docx_to_blocks(path);
        let params = text_segmenter::TextChunkParams::default();
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

    // Excel (encoding ignored)
    if lower.ends_with(".xlsx") || lower.ends_with(".xls") || lower.ends_with(".ods") {
        let blocks: Vec<UnifiedBlock> = reader_excel::read_excel_to_blocks(path);
        let params = text_segmenter::TextChunkParams::default();
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

    // Text-like with encoding hint
    if is_text_like(path) {
        let blocks: Vec<UnifiedBlock> = reader_txt::read_txt_to_blocks_with_encoding(path, encoding);
        let params = text_segmenter::TextChunkParams::default();
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
    chunk_file_with_file_record(path)
}

/// Unified entry with explicit TextChunkParams and optional encoding for text-like files.
pub fn chunk_file_with_file_record_with_params(
    path: &str,
    encoding: Option<&str>,
    params: &text_segmenter::TextChunkParams,
) -> ChunkOutput {
    let lower = path.to_lowercase();
    // PDF via unified text params
    if lower.ends_with(".pdf") {
        let blocks: Vec<UnifiedBlock> = reader_pdf::read_pdf_to_blocks(path);
        let segs = pdf_chunker::chunk_pdf_blocks_to_segments_with_text_params(&blocks, params);
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
    }

    // DOCX via unified text params (hard-cut at top-level headings)
    if lower.ends_with(".docx") {
        let blocks: Vec<UnifiedBlock> = reader_docx::read_docx_to_blocks(path);
        let segs = chunk_blocks_grouped_by_h1(&blocks, params);
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

    // Excel via unified text params (hard-cut at sheet = top-level heading)
    if lower.ends_with(".xlsx") || lower.ends_with(".xls") || lower.ends_with(".ods") {
        let blocks: Vec<UnifiedBlock> = reader_excel::read_excel_to_blocks(path);
        let segs = chunk_blocks_grouped_by_h1(&blocks, params);
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

    // Text-like via unified text params (hard-cut at top-level headings when present)
    if is_text_like(path) {
        let blocks: Vec<UnifiedBlock> = reader_txt::read_txt_to_blocks_with_encoding(path, encoding);
        let segs = chunk_blocks_grouped_by_h1(&blocks, params);
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
    chunk_file_with_file_record(path)
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

