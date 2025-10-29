use std::fs;

use crate::unified_blocks::{UnifiedBlock, BlockKind};

/// Read a UTF-8 (or lossily-decoded) text file and convert it into paragraph blocks.
/// Paragraphs are split on blank lines (two or more consecutive newlines),
/// preserving single newlines within a paragraph.
pub fn read_txt_to_blocks(path: &str) -> Vec<UnifiedBlock> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(_) => return vec![UnifiedBlock::new(BlockKind::Paragraph, "(error) failed to read .txt file", 0, path, "txt")],
    };
    let mut text = String::from_utf8_lossy(&bytes).to_string();
    // Normalize CRLF to LF
    text = text.replace('\r', "");

    // Split into paragraphs by blank lines (two or more newlines)
    let mut out: Vec<UnifiedBlock> = Vec::new();
    let mut order = 0u32;
    let mut start = 0usize;
    let bytes = text.as_bytes();
    let mut i = 0usize;
    let len = bytes.len();
    while i < len {
        if bytes[i] == b'\n' {
            // Count consecutive newlines
            let mut j = i;
            let mut nl = 0usize;
            while j < len && bytes[j] == b'\n' { nl += 1; j += 1; }
            if nl >= 2 {
                // paragraph boundary is [start, i)
                if i > start {
                    let para = text[start..i].trim_matches('\n').trim();
                    if !para.is_empty() {
                        let b = UnifiedBlock::new(BlockKind::Paragraph, para.to_string(), order, path, "txt");
                        out.push(b);
                        order += 1;
                    }
                }
                start = j; // skip the blank lines
                i = j;
                continue;
            }
        }
        i += 1;
    }
    // Trailing paragraph
    if start < len {
        let para = text[start..len].trim_matches('\n').trim();
        if !para.is_empty() {
            let b = UnifiedBlock::new(BlockKind::Paragraph, para.to_string(), order, path, "txt");
            out.push(b);
        }
    }

    if out.is_empty() {
        out.push(UnifiedBlock::new(BlockKind::Paragraph, String::new(), 0, path, "txt"));
    }
    out
}

