pub mod reader_pdf;
pub mod reader_docx;
pub mod reader_txt;
pub mod unified_blocks;
pub mod chunker_rules_jp;
pub mod text_segmenter;
#[cfg(feature = "pdfium")] pub mod reader_pdf_pdfium;
#[cfg(feature = "pure-pdf")] pub mod reader_pdf_pure;
pub mod pdf_chunker;

use chunk_model::{ChunkId, ChunkRecord, DocumentId, FileRecord};
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
    // PDF: delegate to dedicated chunker which already returns page ranges
    if path.ends_with(".pdf") {
        let params = pdf_chunker::PdfChunkParams::default();
        let (file, chunks) = pdf_chunker::chunk_pdf_file_with_file_record(path, &params);
        return ChunkOutput { file, chunks };
    }

    // DOCX: read blocks, segment with text_segmenter (preserve page ranges from blocks/segmenter)
    if path.ends_with(".docx") {
        let blocks: Vec<UnifiedBlock> = reader_docx::read_docx_to_blocks(path);
        let params = text_segmenter::TextChunkParams::default();
        let segs = text_segmenter::chunk_blocks_to_segments(&blocks, &params);

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

        let file = FileRecord {
            schema_version: chunk_model::SCHEMA_MAJOR,
            doc_id: DocumentId(path.to_string()),
            doc_revision: Some(1),
            source_uri: path.to_string(),
            source_mime: "application/vnd.openxmlformats-officedocument.wordprocessingml.document".into(),
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
            reader_backend: Some("docx".into()),
            ocr_used: None,
            ocr_langs: Vec::new(),
            chunk_count: Some(chunks.len() as u32),
            total_tokens: None,
            meta: BTreeMap::new(),
            extra: BTreeMap::new(),
        };
        return ChunkOutput { file, chunks };
    }

    // Text-like files: segment and set page to 1
    if is_text_like(path) {
        let blocks: Vec<UnifiedBlock> = reader_txt::read_txt_to_blocks(path);
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

        let file = FileRecord {
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
            reader_backend: Some("txt".into()),
            ocr_used: None,
            ocr_langs: Vec::new(),
            chunk_count: Some(chunks.len() as u32),
            total_tokens: None,
            meta: BTreeMap::new(),
            extra: BTreeMap::new(),
        };
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

    let file = FileRecord {
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
    ChunkOutput { file, chunks }
}

/// Variant with an explicit encoding hint for text-like files.
/// For non-text formats (PDF/DOCX), the behavior is identical to `chunk_file_with_file_record`.
pub fn chunk_file_with_file_record_with_encoding(path: &str, encoding: Option<&str>) -> ChunkOutput {
    // PDF
    if path.ends_with(".pdf") {
        let params = pdf_chunker::PdfChunkParams::default();
        let (file, chunks) = pdf_chunker::chunk_pdf_file_with_file_record(path, &params);
        return ChunkOutput { file, chunks };
    }

    // DOCX
    if path.ends_with(".docx") {
        let blocks: Vec<UnifiedBlock> = reader_docx::read_docx_to_blocks(path);
        let params = text_segmenter::TextChunkParams::default();
        let segs = text_segmenter::chunk_blocks_to_segments(&blocks, &params);

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

        let file = FileRecord {
            schema_version: chunk_model::SCHEMA_MAJOR,
            doc_id: DocumentId(path.to_string()),
            doc_revision: Some(1),
            source_uri: path.to_string(),
            source_mime: "application/vnd.openxmlformats-officedocument.wordprocessingml.document".into(),
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
            reader_backend: Some("docx".into()),
            ocr_used: None,
            ocr_langs: Vec::new(),
            chunk_count: Some(chunks.len() as u32),
            total_tokens: None,
            meta: BTreeMap::new(),
            extra: BTreeMap::new(),
        };
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

        let file = FileRecord {
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
            reader_backend: Some("txt".into()),
            ocr_used: None,
            ocr_langs: Vec::new(),
            chunk_count: Some(chunks.len() as u32),
            total_tokens: None,
            meta: BTreeMap::new(),
            extra: BTreeMap::new(),
        };
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

