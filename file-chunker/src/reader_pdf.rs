use crate::unified_blocks::UnifiedBlock;

/// Stub PDF reader that returns a single block placeholder.
pub fn read_pdf_to_blocks(_path: &str) -> Vec<UnifiedBlock> {
    vec![UnifiedBlock::new("(stub) extracted text from PDF")]
}

