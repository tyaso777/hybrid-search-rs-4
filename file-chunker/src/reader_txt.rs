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


/// Read a text file with an optional explicit encoding and convert to paragraph blocks.
/// Supported encodings: "utf-8" (default), "shift_jis" (aliases: "sjis", "cp932", "windows-31j"),
/// "windows-1252", "utf-16le", "utf-16be". Unknown values fall back to UTF-8 (lossy).
pub fn read_txt_to_blocks_with_encoding(path: &str, encoding: Option<&str>) -> Vec<UnifiedBlock> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(_) => return vec![UnifiedBlock::new(BlockKind::Paragraph, "(error) failed to read .txt file", 0, path, "txt")],
    };

    let lower = encoding.unwrap_or("").to_ascii_lowercase();
    let mut text: String = match lower.as_str() {
        "utf-8" | "utf8" | "" => String::from_utf8_lossy(&bytes).to_string(),
        "shift_jis" | "sjis" | "cp932" | "windows-31j" => {
            let (cow, _enc_used, _had_errors) = encoding_rs::SHIFT_JIS.decode(&bytes);
            cow.into_owned()
        }
        "windows-1252" | "cp1252" => {
            let (cow, _enc_used, _had_errors) = encoding_rs::WINDOWS_1252.decode(&bytes);
            cow.into_owned()
        }
        "utf-16le" | "utf16le" => {
            let mut u16s: Vec<u16> = Vec::with_capacity(bytes.len() / 2);
            let mut i = 0usize;
            // Skip BOM if present
            if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE { i = 2; }
            while i + 1 < bytes.len() { u16s.push(u16::from_le_bytes([bytes[i], bytes[i + 1]])); i += 2; }
            String::from_utf16_lossy(&u16s)
        }
        "utf-16be" | "utf16be" => {
            let mut u16s: Vec<u16> = Vec::with_capacity(bytes.len() / 2);
            let mut i = 0usize;
            // Skip BOM if present
            if bytes.len() >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF { i = 2; }
            while i + 1 < bytes.len() { u16s.push(u16::from_be_bytes([bytes[i], bytes[i + 1]])); i += 2; }
            String::from_utf16_lossy(&u16s)
        }
        _ => String::from_utf8_lossy(&bytes).to_string(),
    };

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
