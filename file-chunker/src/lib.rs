pub mod reader_pdf;
pub mod reader_docx;
pub mod unified_blocks;
pub mod chunker_rules_jp;

use chunk_model::{ChunkId, ChunkRecord, DocumentId};
use std::collections::BTreeMap;
use unified_blocks::UnifiedBlock;

/// High-level entry to chunk a file by path.
/// This is a stubbed pipeline that returns a single trivial chunk for now.
pub fn chunk_file(path: &str) -> Vec<ChunkRecord> {
    // Simple content-type inference by extension (stub)
    let content_type = if path.ends_with(".pdf") {
        "application/pdf"
    } else if path.ends_with(".docx") {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    } else {
        "text/plain"
    };

    // Read into unified blocks (stubbed per type)
    let blocks: Vec<UnifiedBlock> = match content_type {
        "application/pdf" => reader_pdf::read_pdf_to_blocks(path),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            reader_docx::read_docx_to_blocks(path)
        }
        _ => vec![UnifiedBlock::new("(stub) read file content here")],
    };

    // Apply JP chunking rules (stub)
    let chunks_text = chunker_rules_jp::chunk_blocks_jp(&blocks);

    // Map to ChunkRecord (schema-based)
    chunks_text
        .into_iter()
        .enumerate()
        .map(|(i, text)| {
            ChunkRecord {
                schema_version: chunk_model::SCHEMA_MAJOR,
                doc_id: DocumentId(path.to_string()),
                chunk_id: ChunkId(format!("{}#{}", path, i)),
                source_uri: path.to_string(),
                source_mime: content_type.to_string(),
                extracted_at: String::new(), // optionally fill with ISO 8601 UTC
                text,
                section_path: Vec::new(),
                meta: BTreeMap::new(),
                extra: BTreeMap::new(),
            }
        })
        .collect()
}

