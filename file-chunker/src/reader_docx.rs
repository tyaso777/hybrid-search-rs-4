use crate::unified_blocks::{UnifiedBlock, BlockKind};

/// Stub DOCX reader that returns a single block placeholder.
pub fn read_docx_to_blocks(path: &str) -> Vec<UnifiedBlock> {
    let mut b = UnifiedBlock::new(BlockKind::Paragraph, "(stub) extracted text from DOCX", 0, path, "stub.docx");
    b.page_start = None;
    b.page_end = None;
    vec![b]
}
