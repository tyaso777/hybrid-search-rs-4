use crate::unified_blocks::UnifiedBlock;

/// Stub DOCX reader that returns a single block placeholder.
pub fn read_docx_to_blocks(_path: &str) -> Vec<UnifiedBlock> {
    vec![UnifiedBlock::new("(stub) extracted text from DOCX")]
}

