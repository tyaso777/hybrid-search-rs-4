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

/// Result bundle including file-level metadata and chunk list.
#[derive(Debug, Clone)]
pub struct ChunkOutput {
    pub file: FileRecord,
    pub chunks: Vec<ChunkRecord>,
}

/// High-level entry to chunk a file by path and return file/chunks.
/// This is a stubbed pipeline that returns a simple chunking for now.
pub fn chunk_file_with_file_record(path: &str) -> ChunkOutput {
    // Simple content-type inference by extension (stub)
    let content_type = if path.ends_with(".pdf") {
        "application/pdf"
    } else if path.ends_with(".docx") {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    } else {
        "text/plain"
    };

    // Read and chunk by type (PDF uses dedicated chunker with heuristics)
    let chunks_text: Vec<String> = match content_type {
        "application/pdf" => {
            let params = pdf_chunker::PdfChunkParams::default();
            let (_file, chunks) = pdf_chunker::chunk_pdf_file_with_file_record(path, &params);
            chunks.into_iter().map(|c| c.text).collect()
        }
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            let blocks: Vec<UnifiedBlock> = reader_docx::read_docx_to_blocks(path);
            chunker_rules_jp::chunk_blocks_jp(&blocks)
        }
        _ => {
            if path.ends_with(".txt") {
                let blocks: Vec<UnifiedBlock> = reader_txt::read_txt_to_blocks(path);
                let params = text_segmenter::TextChunkParams::default();
                let segs = text_segmenter::chunk_blocks_to_segments(&blocks, &params);
                segs.into_iter().map(|(t, _, _)| t).collect()
            } else {
                let blocks: Vec<UnifiedBlock> = vec![UnifiedBlock::new(BlockKind::Paragraph, "(stub) read file content here", 0, path, "stub.plain")];
                chunker_rules_jp::chunk_blocks_jp(&blocks)
            }
        }
    };

    // Build ChunkRecord list (schema-based)
    let chunks: Vec<ChunkRecord> = chunks_text
        .into_iter()
        .enumerate()
        .map(|(i, text)| ChunkRecord {
            schema_version: chunk_model::SCHEMA_MAJOR,
            doc_id: DocumentId(path.to_string()),
            chunk_id: ChunkId(format!("{}#{}", path, i)),
            source_uri: path.to_string(),
            source_mime: content_type.to_string(),
            extracted_at: String::new(),
            page_start: None,
            page_end: None,
            text,
            section_path: None,
            meta: BTreeMap::new(),
            extra: BTreeMap::new(),
        })
        .collect();

    // Construct FileRecord
    let file = FileRecord {
        schema_version: chunk_model::SCHEMA_MAJOR,
        doc_id: DocumentId(path.to_string()),
        doc_revision: Some(1),
        source_uri: path.to_string(),
        source_mime: content_type.to_string(),
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
        reader_backend: None,
        ocr_used: None,
        ocr_langs: Vec::new(),
        chunk_count: Some(chunks.len() as u32),
        total_tokens: None,
        meta: BTreeMap::new(),
        extra: BTreeMap::new(),
    };

    ChunkOutput { file, chunks }
}

/// Legacy helper returning only chunks for backward compatibility.
pub fn chunk_file(path: &str) -> Vec<ChunkRecord> {
    chunk_file_with_file_record(path).chunks
}

