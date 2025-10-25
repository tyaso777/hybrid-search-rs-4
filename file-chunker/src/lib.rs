pub mod reader_pdf;
pub mod reader_docx;
pub mod unified_blocks;
pub mod chunker_rules_jp;

use chunk_model::{ChunkRecord, SourceMeta};
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

    // Map to ChunkRecord
    let source = SourceMeta::new(path, content_type);
    chunks_text
        .into_iter()
        .enumerate()
        .map(|(i, text)| {
            let mut cr = ChunkRecord::new(format!("{}#{}", path, i), source.clone(), i as u32, text);
            cr.tokens = cr.text.chars().count(); // naive token estimate
            cr
        })
        .collect()
}

