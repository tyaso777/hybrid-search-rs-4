use crate::unified_blocks::UnifiedBlock;

/// Very simple JP-ish chunking: split blocks by double newline or full stop-like punctuation.
/// This is a stub and should be replaced with real rules later.
pub fn chunk_blocks_jp(blocks: &[UnifiedBlock]) -> Vec<String> {
    let mut out = Vec::new();
    for b in blocks {
        // naive split on Japanese/English sentence delimiters
        let text = b.text.replace('\r', "");
        let mut start = 0usize;
        for (idx, ch) in text.char_indices() {
            if matches!(ch, '。' | '！' | '？' | '.' | '!' | '?') {
                let end = idx + ch.len_utf8();
                let seg = text[start..end].trim();
                if !seg.is_empty() {
                    out.push(seg.to_string());
                }
                start = end;
            }
        }
        let tail = text[start..].trim();
        if !tail.is_empty() {
            out.push(tail.to_string());
        }
    }
    if out.is_empty() {
        out.push("".to_string());
    }
    out
}

