//! PDFium-backed PDF reader. Behind feature `pdfium`.

#![cfg(feature = "pdfium")]

use crate::unified_blocks::{UnifiedBlock, BlockKind};
use pdfium_render::prelude::*;
use std::path::PathBuf;

fn bind_pdfium_from_env() -> Option<Box<dyn PdfiumLibraryBindings>> {
    // Prefer explicit full DLL path
    if let Ok(path) = std::env::var("PDFIUM_DLL_PATH") {
        let pb = PathBuf::from(path);
        let lib_path = if pb.is_dir() {
            Pdfium::pdfium_platform_library_name_at_path(&pb)
        } else {
            pb
        };
        if let Ok(b) = Pdfium::bind_to_library(&lib_path) { return Some(b); }
    }
    // Common alternative env var that points to a dir
    if let Ok(dir) = std::env::var("PDFIUM_DIR") {
        let pb = PathBuf::from(dir);
        let lib_path = Pdfium::pdfium_platform_library_name_at_path(&pb);
        if let Ok(b) = Pdfium::bind_to_library(&lib_path) { return Some(b); }
    }
    None
}

fn bind_pdfium_from_bundle() -> Option<Box<dyn PdfiumLibraryBindings>> {
    // Resolve relative to file-chunker crate root
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bin");
    let arch = base.join("pdfium-win-x64");
    let cand1: PathBuf = Pdfium::pdfium_platform_library_name_at_path(&arch.join("bin"));
    let cand2: PathBuf = Pdfium::pdfium_platform_library_name_at_path(&arch);
    let cand3: PathBuf = Pdfium::pdfium_platform_library_name_at_path(&base);
    let candidates: [PathBuf; 3] = [cand1, cand2, cand3];
    for p in &candidates {
        if p.exists() {
            if let Ok(b) = Pdfium::bind_to_library(p) { return Some(b); }
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfStructureMode {
    /// Return a single paragraph block per page after normalization.
    Plain,
    /// Heuristic segmentation into headings / list items / paragraphs from normalized text.
    Heuristic,
    /// Try Heuristic, but fall back to Plain if no signal is detected.
    Auto,
}

pub fn read_pdf_to_blocks_pdfium(path: &str) -> Vec<UnifiedBlock> {
    read_pdf_to_blocks_pdfium_with_mode(path, PdfStructureMode::Auto)
}

pub fn read_pdf_to_blocks_pdfium_with_mode(path: &str, mode: PdfStructureMode) -> Vec<UnifiedBlock> {
    // Try env override → bundled under file-chunker/bin → system library
    let bindings = if let Some(b) = bind_pdfium_from_env() {
        b
    } else if let Some(b) = bind_pdfium_from_bundle() {
        b
    } else {
        match Pdfium::bind_to_system_library() {
            Ok(b) => b,
            Err(err) => {
                return vec![UnifiedBlock::new(
                    BlockKind::Paragraph,
                    format!("[pdfium] failed to bind: {}", err),
                    0,
                    path,
                    "pdfium",
                )];
            }
        }
    };

    let pdfium = Pdfium::new(bindings);
    let document = match pdfium.load_pdf_from_file(path, None) {
        Ok(d) => d,
        Err(err) => {
            return vec![UnifiedBlock::new(
                BlockKind::Paragraph,
                format!("[pdfium] failed to open PDF: {}", err),
                0,
                path,
                "pdfium",
            )];
        }
    };

    let mut out = Vec::new();
    let mut order = 0u32;
    for (idx, page) in document.pages().iter().enumerate() {
        let page_num = (idx as u32) + 1;
        let text = match page.text() {
            Ok(t) => t.all(),
            Err(_) => String::new(),
        };
        let text = normalize_pdfium_text(&text);
        if text.trim().is_empty() { continue; }

        match mode {
            PdfStructureMode::Plain => {
                let mut block = UnifiedBlock::new(
                    BlockKind::Paragraph,
                    text,
                    order,
                    path,
                    "pdfium",
                );
                order += 1;
                block.page_start = Some(page_num);
                block.page_end = Some(page_num);
                out.push(block);
            }
            PdfStructureMode::Heuristic => {
                let created = push_heuristic_blocks(&mut out, &text, path, page_num, &mut order);
                if created == 0 {
                    // Fallback safety: ensure at least one block per page
                    let mut block = UnifiedBlock::new(
                        BlockKind::Paragraph,
                        text,
                        order,
                        path,
                        "pdfium",
                    );
                    order += 1;
                    block.page_start = Some(page_num);
                    block.page_end = Some(page_num);
                    out.push(block);
                }
            }
            PdfStructureMode::Auto => {
                // Perform heuristic segmentation but decide whether to keep it.
                let _created = push_heuristic_blocks(&mut out, &text, path, page_num, &mut order);
                // If segmentation produced only a single long paragraph and no headings/lists,
                // consider it "unknown structure" and collapse into one block.
                let page_blocks = out.iter().rev().take_while(|b| b.page_start == Some(page_num)).count();
                let has_structure = out.iter().rev().take(page_blocks)
                    .any(|b| matches!(b.kind, BlockKind::Heading | BlockKind::ListItem));
                if !has_structure {
                    // Remove the blocks we just added for this page and add a plain block instead.
                    for _ in 0..page_blocks { let _ = out.pop(); }
                    let mut block = UnifiedBlock::new(
                        BlockKind::Paragraph,
                        text,
                        order,
                        path,
                        "pdfium",
                    );
                    order += 1;
                    block.page_start = Some(page_num);
                    block.page_end = Some(page_num);
                    out.push(block);
                }
            }
        }
    }

    out
}

// --- Text normalization -----------------------------------------------------

fn normalize_pdfium_text(raw: &str) -> String {
    // 1) Normalize CRLF to LF
    let raw = raw.replace('\r', "");

    // 2) Split to lines and prune page header/footer like "- 12 -" at the very top/bottom.
    let lines: Vec<&str> = raw.split('\n').collect();
    let header_idx = first_nonblank_index(&lines).and_then(|i|
        if is_decorated_page_number_line(lines[i])
            || is_page_label_line(lines[i])
            || is_number_pair_line(lines[i])
            || is_digits_of_digits_line(lines[i]) {
            Some(i)
        } else { None }
    );
    let footer_idx = last_nonblank_index(&lines).and_then(|i|
        if is_decorated_page_number_line(lines[i])
            || is_page_label_line(lines[i])
            || is_number_pair_line(lines[i])
            || is_digits_of_digits_line(lines[i]) {
            Some(i)
        } else { None }
    );

    // 2b) Pre-scan visual widths, excluding detected header/footer for max length estimation.
    let lens: Vec<f32> = lines.iter().map(|l| visual_len(l.trim_end())).collect();
    let mut max_len = 0.0f32;
    for (i, w) in lens.iter().enumerate() {
        let is_pruned = Some(i) == header_idx || Some(i) == footer_idx;
        if !is_pruned { max_len = max_len.max(*w); }
    }
    // "Two characters" threshold in visual units, adapted to script mix (ASCII vs CJK).
    // This makes us conservative: we only keep an explicit line break when the previous
    // line ends at least ~2 characters left of the common right edge. Otherwise, we join.
    let char_gap_threshold = 2.0 * avg_char_unit(&raw);

    // 3) Build output with heuristics:
    //    - Blank line => paragraph separator (\n\n)
    //    - Join by default; only keep a visible line break when the previous line ends
    //      >= ~2 characters left of the common right edge (conservative break detection)
    //    - Hyphenation join: remove trailing hyphen when next starts with ASCII alpha
    //    - ASCII<->ASCII joins insert a single space; CJK joins insert none

    let mut out = String::with_capacity(raw.len());
    let mut started = false;
    let mut prev_w = 0.0f32;
    let mut prev_ended_with_hyphen = false;

    for (i, raw_line) in lines.iter().enumerate() {
        let is_pruned_hf = Some(i) == header_idx || Some(i) == footer_idx;
        let l = raw_line.trim_end();
        let is_blank = l.trim().is_empty();
        if is_pruned_hf { continue; }
        let w = lens.get(i).copied().unwrap_or(0.0);

        if is_blank {
            if started {
                if !out.ends_with("\n\n") {
                    if !out.ends_with('\n') { out.push('\n'); }
                    out.push('\n');
                }
                prev_w = 0.0;
                prev_ended_with_hyphen = false;
            }
            continue;
        }

        if looks_like_heading_or_list(l) && started && !out.ends_with("\n\n") {
            if !out.ends_with('\n') { out.push('\n'); }
            out.push('\n');
        }

        if started && !out.ends_with("\n\n") {
            // How far the previous line's end is from the estimated right edge.
            let short_by = (max_len - prev_w).max(0.0);
            // Join unless the previous line is clearly short (>= ~2 chars left).
            let should_join_by_width = short_by < char_gap_threshold;

            if prev_ended_with_hyphen && first_non_ws_is_ascii_alpha(l) {
                pop_trailing_hyphen(&mut out);
                out.push_str(l.trim_start());
            } else if should_join_by_width {
                if last_non_ws_is_ascii_alnum(&out) && first_non_ws_is_ascii_alnum(l) { out.push(' '); }
                out.push_str(l.trim_start());
            } else {
                out.push('\n');
                out.push_str(l.trim_start());
            }
        } else {
            out.push_str(l.trim_start());
            started = true;
        }

        prev_ended_with_hyphen = ends_with_hyphen_like(l);
        prev_w = w;
    }

    out
}

// --- Heuristic segmentation --------------------------------------------------

fn push_heuristic_blocks(out: &mut Vec<UnifiedBlock>, text: &str, origin: &str, page_num: u32, order: &mut u32) -> usize {
    let base_len = out.len();
    let mut para_buf = String::new();

    let flush_para = |buf: &mut String, out: &mut Vec<UnifiedBlock>, order: &mut u32| {
        if buf.trim().is_empty() { return; }
        let mut b = UnifiedBlock::new(BlockKind::Paragraph, buf.trim(), *order, origin, "pdfium");
        *order += 1;
        b.page_start = Some(page_num);
        b.page_end = Some(page_num);
        out.push(b);
        buf.clear();
    };

    for line in text.split('\n') {
        let l = line.trim_end();
        if l.trim().is_empty() {
            flush_para(&mut para_buf, out, order);
            continue;
        }

        if let Some(level) = heading_level_guess(l) {
            flush_para(&mut para_buf, out, order);
            let mut b = UnifiedBlock::new(BlockKind::Heading, l.trim(), *order, origin, "pdfium");
            *order += 1;
            b.page_start = Some(page_num);
            b.page_end = Some(page_num);
            b.heading_level = Some(level);
            b.section_hint = Some(crate::unified_blocks::SectionHint { level, title: l.trim().to_string(), numbering: None });
            out.push(b);
            continue;
        }

        if let Some(list_info) = list_info_guess(l) {
            flush_para(&mut para_buf, out, order);
            let mut b = UnifiedBlock::new(BlockKind::ListItem, l.trim(), *order, origin, "pdfium");
            *order += 1;
            b.page_start = Some(page_num);
            b.page_end = Some(page_num);
            b.list = Some(list_info);
            out.push(b);
            continue;
        }

        // Accumulate into current paragraph
        if !para_buf.is_empty() && last_non_ws_is_ascii_alnum(&para_buf) && first_non_ws_is_ascii_alnum(l) {
            para_buf.push(' ');
        }
        para_buf.push_str(l.trim_start());
    }

    flush_para(&mut para_buf, out, order);
    out.len().saturating_sub(base_len)
}

fn heading_level_guess(line: &str) -> Option<u8> {
    let s = line.trim_start();
    // 第n章 → level 1
    if s.starts_with('第') && s.contains('章') {
        return Some(1);
    }
    // n.  / n)  / n.n  patterns → level by dot count (capped)
    let mut digits = 0usize;
    let mut dots = 0usize;
    for ch in s.chars().take(12) {
        if ch.is_ascii_digit() { digits += 1; continue; }
        if ch == '.' { dots += 1; continue; }
        if ch == ')' && digits > 0 { break; }
        if ch.is_whitespace() { break; }
        // Non-matching char early
        break;
    }
    if digits > 0 {
        let level = (dots as u8).saturating_add(1).min(6);
        return Some(level);
    }
    None
}

fn list_info_guess(line: &str) -> Option<crate::unified_blocks::ListInfo> {
    let s = line.trim_start();
    // Bulleted
    if s.starts_with('・') || s.starts_with('•') || s.starts_with("-") {
        return Some(crate::unified_blocks::ListInfo { ordered: false, level: 1, marker: s.chars().next().map(|c| c.to_string()) });
    }
    // Ordered: 1. foo  or  1) foo
    let mut i = 0usize;
    let chars: Vec<char> = s.chars().collect();
    while i < chars.len() && chars[i].is_ascii_digit() { i += 1; }
    if i > 0 && i < chars.len() {
        if chars[i] == '.' || chars[i] == ')' {
            return Some(crate::unified_blocks::ListInfo { ordered: true, level: 1, marker: Some(chars[i].to_string()) });
        }
    }
    None
}

fn visual_len(s: &str) -> f32 {
    let mut w = 0.0f32;
    for ch in s.chars() {
        if ch.is_whitespace() { continue; }
        if ch.is_ascii() { w += 0.55; } else { w += 1.0; }
    }
    w
}

// Average visual width per non-whitespace character across the whole text.
// ASCII letters count narrower (~0.55), CJK as 1.0. Used to scale the
// "two characters" threshold to the script mix of the page.
fn avg_char_unit(s: &str) -> f32 {
    let mut units = 0.0f32;
    let mut count = 0u32;
    for ch in s.chars() {
        if ch.is_whitespace() { continue; }
        units += if ch.is_ascii() { 0.55 } else { 1.0 };
        count += 1;
    }
    if count == 0 { 1.0 } else { units / (count as f32) }
}

// Identify first/last non-blank line indices.
fn first_nonblank_index(lines: &[&str]) -> Option<usize> {
    for (i, l) in lines.iter().enumerate() {
        if !l.trim().is_empty() { return Some(i); }
    }
    None
}

fn last_nonblank_index(lines: &[&str]) -> Option<usize> {
    for (i, l) in lines.iter().enumerate().rev() {
        if !l.trim().is_empty() { return Some(i); }
    }
    None
}

// Detect page number lines like "- 12 -", "(3)", "★5☆" with any number of ornaments on each side.
fn is_decorated_page_number_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() { return false; }
    // Remove all whitespace to tolerate spaces like "- 1 -"
    let compact: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.is_empty() { return false; }

    let chars: Vec<char> = compact.chars().collect();
    let n = chars.len();
    let mut i = 0usize;

    // Zero or more leading ornaments
    while i < n && is_ornament_char(chars[i]) { i += 1; }

    // One or more digits (ASCII or full-width)
    let mut digits = 0usize;
    while i < n && is_ascii_or_fullwidth_digit(chars[i]) { i += 1; digits += 1; }
    if digits == 0 { return false; }

    // Zero or more trailing ornaments and nothing else
    while i < n && is_ornament_char(chars[i]) { i += 1; }

    i == n
}

// Detect page header/footer lines containing page labels like "Page 3", "p. 12",
// "pp. 12–14", "3ページ", "第5頁". Requires that the line contains a page keyword and a digit
// (ASCII or full-width). Whitespace is tolerated; only evaluated for first/last lines.
fn is_page_label_line(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() { return false; }
    let ascii_lower = t.to_lowercase();
    let has_digit = t.chars().any(is_ascii_or_fullwidth_digit);
    if !has_digit { return false; }

    if ascii_lower.contains("page") { return true; }
    if ascii_lower.contains("p.") || ascii_lower.contains("p．") { return true; }
    if contains_pp_label(&ascii_lower) { return true; }
    if t.contains("ページ") || t.contains("頁") { return true; }
    false
}

// Rough detection for plural page label "pp".
// Matches: "pp.", "pp ", "pp12", "pp-12", "pp/12", "pp〜12" etc.
fn contains_pp_label(s_lower: &str) -> bool {
    let chars: Vec<char> = s_lower.chars().collect();
    let n = chars.len();
    let is_follow_ok = |c: char| -> bool {
        c.is_ascii_digit() || c.is_whitespace() || c == '.' || c == '．' || is_pair_sep_char(c)
    };
    let mut i = 0usize;
    while i + 1 < n {
        if chars[i] == 'p' && chars[i + 1] == 'p' {
            if i + 2 >= n { return true; }
            let c = chars[i + 2];
            if is_follow_ok(c) { return true; }
        }
        i += 1;
    }
    false
}

// Detect compact numeric pair like "12/34", "3-4", allowing any number of
// ornaments on both sides. The entire line must be composed of ornaments,
// digits, and a separator ('/' or '-' family), no other letters.
fn is_number_pair_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() { return false; }
    let compact: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
    if compact.is_empty() { return false; }
    let chars: Vec<char> = compact.chars().collect();
    let n = chars.len();
    let mut i = 0usize;

    // leading ornaments
    while i < n && is_ornament_char(chars[i]) { i += 1; }

    // first number
    let mut d1 = 0usize;
    while i < n && is_ascii_or_fullwidth_digit(chars[i]) { i += 1; d1 += 1; }
    if d1 == 0 { return false; }

    // one or more separators
    let mut sep = 0usize;
    while i < n && is_pair_sep_char(chars[i]) { i += 1; sep += 1; }
    if sep == 0 { return false; }

    // second number
    let mut d2 = 0usize;
    while i < n && is_ascii_or_fullwidth_digit(chars[i]) { i += 1; d2 += 1; }
    if d2 == 0 { return false; }

    // trailing ornaments
    while i < n && is_ornament_char(chars[i]) { i += 1; }

    i == n
}

// Detect lines like "1 of 12" (case-insensitive), allowing spaces and ornaments around.
fn is_digits_of_digits_line(line: &str) -> bool {
    let t = line.trim();
    if t.is_empty() { return false; }
    // Strip ornaments at both ends and collapse spaces to simplify matching
    let s: String = t.chars().filter(|c| !is_ornament_char(*c)).collect();
    let s = s.trim();
    // Normalize internal whitespace
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    let s_lower = s.to_lowercase();

    // Fast path: must contain " of "
    if !s_lower.contains(" of ") { return false; }

    // Check digits on both sides (accept full-width digits too)
    let parts: Vec<&str> = s_lower.split(" of ").collect();
    if parts.len() != 2 { return false; }
    let left_has_digit = parts[0].chars().any(is_ascii_or_fullwidth_digit) && parts[0].chars().all(|c| c.is_whitespace() || is_ascii_or_fullwidth_digit(c));
    let right_has_digit = parts[1].chars().any(is_ascii_or_fullwidth_digit) && parts[1].chars().all(|c| c.is_whitespace() || is_ascii_or_fullwidth_digit(c));
    left_has_digit && right_has_digit
}

fn is_pair_sep_char(c: char) -> bool {
    matches!(c, '/' | '／' | '-' | '‐' | '‑' | '‒' | '–' | '—' | '−' | '~' | '〜' | '～')
}

fn is_ascii_or_fullwidth_digit(c: char) -> bool {
    c.is_ascii_digit() || ('０'..='９').contains(&c)
}

fn is_ornament_char(c: char) -> bool {
    match c {
        // dashes/lines
        '-' | '‐' | '‑' | '‒' | '–' | '—' | '―' | '−' | '─' | '━' | '_'
        // bullets/stars/shapes
        | '*' | '•' | '·' | '・' | '●' | '○' | '◇' | '◆' | '☆' | '★'
        // brackets/angles/quotes
        | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | '‹' | '›' | '«' | '»'
        | '「' | '」' | '『' | '』' | '【' | '】' | '〈' | '〉' | '《' | '》'
        // misc decorations
        | '|' | '｜' | '=' | '~' | '〜' | '～' | '†' | '‡' | '§' | '¶' | '※'
            => true,
        _ => false,
    }
}

fn ends_with_hyphen_like(s: &str) -> bool {
    s.trim_end().ends_with('-') || s.trim_end().ends_with('‐') || s.trim_end().ends_with('−')
}

fn pop_trailing_hyphen(buf: &mut String) {
    while buf.ends_with(' ') { buf.pop(); }
    if buf.ends_with('−') || buf.ends_with('‐') || buf.ends_with('-') { buf.pop(); }
}

fn first_non_ws_is_ascii_alpha(s: &str) -> bool {
    s.chars().find(|c| !c.is_whitespace()).map(|c| c.is_ascii_alphabetic()).unwrap_or(false)
}

fn first_non_ws_is_ascii_alnum(s: &str) -> bool {
    s.chars().find(|c| !c.is_whitespace()).map(|c| c.is_ascii_alphanumeric()).unwrap_or(false)
}

fn last_non_ws_is_ascii_alnum(s: &str) -> bool {
    s.chars().rev().find(|c| !c.is_whitespace()).map(|c| c.is_ascii_alphanumeric()).unwrap_or(false)
}

fn looks_like_heading_or_list(s: &str) -> bool {
    let st = s.trim_start();
    // Heading like "第2章" / "第 2 章"
    if st.starts_with('第') && st.contains('章') { return true; }
    // Simple numbered headings: "1.", "1)" at start
    if st.chars().take(3).any(|c| c.is_ascii_digit()) {
        if st.contains('.') || st.contains(')') { return true; }
    }
    // Bullet-like markers
    if st.starts_with('・') || st.starts_with('•') || st.starts_with("-") { return true; }
    false
}
