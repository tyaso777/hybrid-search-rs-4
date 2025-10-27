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

pub fn read_pdf_to_blocks_pdfium(path: &str) -> Vec<UnifiedBlock> {
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
        if text.trim().is_empty() {
            continue;
        }

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
        // Page dimensions (optional): can be filled if numeric conversion is desired
        // let w = page.width();
        // let h = page.height();
        // block.bbox = Some(BBox { x: 0.0, y: 0.0, w: w.0, h: h.0, unit: BBoxUnit::Pt });
        out.push(block);
    }

    out
}

// --- Text normalization -----------------------------------------------------

fn normalize_pdfium_text(raw: &str) -> String {
    // 1) Normalize CRLF to LF
    let raw = raw.replace('\r', "");

    // 2) Pre-scan lines to compute a per-page maximum "end position" proxy (visual width).
    let lines: Vec<&str> = raw.split('\n').collect();
    let lens: Vec<f32> = lines.iter().map(|l| visual_len(l.trim_end())).collect();
    let max_len = lens.iter().copied().fold(0.0f32, f32::max);

    // 3) Build output with heuristics:
    //    - Blank line => paragraph separator (\n\n)
    //    - If previous non-blank line nearly reached the page right edge (width ~= max), treat as soft-wrap (join)
    //      otherwise, keep a visible line break (single \n)
    //    - Hyphenation join: remove trailing hyphen when next starts with ASCII alpha
    //    - ASCII<->ASCII joins insert a single space; CJK joins insert none

    let mut out = String::with_capacity(raw.len());
    let mut started = false;
    let mut prev_w = 0.0f32;
    let mut prev_ended_with_hyphen = false;

    for (i, raw_line) in lines.iter().enumerate() {
        let l = raw_line.trim_end();
        let is_blank = l.trim().is_empty();
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
            let near_max = prev_w >= (max_len * 0.97);

            if prev_ended_with_hyphen && first_non_ws_is_ascii_alpha(l) {
                pop_trailing_hyphen(&mut out);
                out.push_str(l.trim_start());
            } else if near_max {
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

fn visual_len(s: &str) -> f32 {
    let mut w = 0.0f32;
    for ch in s.chars() {
        if ch.is_whitespace() { continue; }
        if ch.is_ascii() { w += 0.55; } else { w += 1.0; }
    }
    w
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
