//! Pure-Rust PDF reader (placeholder). Behind feature `pure-pdf`.

#![cfg(feature = "pure-pdf")]

use crate::unified_blocks::{UnifiedBlock, BlockKind};
use lopdf::Document;

/// Minimal pure-Rust PDF reader using `lopdf`.
/// Currently returns one block per page with placeholder text.
pub fn read_pdf_to_blocks_pure(path: &str) -> Vec<UnifiedBlock> {
    let mut out = Vec::new();
    match Document::load(path) {
        Ok(doc) => {
            let pages = doc.get_pages();
            let mut order = 0u32;
            for (page_num, _page_id) in pages.into_iter() {
                let mut block = UnifiedBlock::new(
                    BlockKind::Paragraph,
                    format!("[pure-pdf] page {} (text extraction WIP)", page_num),
                    order,
                    path,
                    "lopdf",
                );
                order += 1;
                block.page_start = Some(page_num);
                block.page_end = Some(page_num);
                out.push(block);
            }
        }
        Err(err) => {
            out.push(UnifiedBlock::new(
                BlockKind::Paragraph,
                format!("[pure-pdf] failed to read: {}", err),
                0,
                path,
                "lopdf",
            ));
        }
    }
    out
}
