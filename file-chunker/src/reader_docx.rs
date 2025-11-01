use crate::unified_blocks::{UnifiedBlock, BlockKind};
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::fs::File;
use std::io::Read;

fn local_name<'a>(q: &'a [u8]) -> &'a [u8] {
    match q.iter().position(|&b| b == b':') { Some(i) => &q[i + 1..], None => q }
}

fn attr_val(e: &BytesStart<'_>, key_local: &[u8]) -> Option<String> {
    for a in e.attributes().with_checks(false) {
        if let Ok(attr) = a {
            let k = local_name(attr.key.as_ref());
            if k == key_local {
                return Some(String::from_utf8_lossy(&attr.value).into_owned());
            }
        }
    }
    None
}

/// Minimal DOCX reader: opens the zip, parses word/document.xml, extracts paragraph text and heading level.
pub fn read_docx_to_blocks(path: &str) -> Vec<UnifiedBlock> {
    let mut blocks: Vec<UnifiedBlock> = Vec::new();
    let file = match File::open(path) { Ok(f) => f, Err(_) => return vec![UnifiedBlock::new(BlockKind::Paragraph, "(error) failed to open DOCX", 0, path, "docx")] };
    let mut zip = match zip::ZipArchive::new(file) { Ok(z) => z, Err(_) => return vec![UnifiedBlock::new(BlockKind::Paragraph, "(error) not a valid .docx (zip) file", 0, path, "docx")] };
    let mut doc_xml = String::new();
    match zip.by_name("word/document.xml") {
        Ok(mut f) => { let _ = f.read_to_string(&mut doc_xml); }
        Err(_) => return vec![UnifiedBlock::new(BlockKind::Paragraph, "(error) missing word/document.xml", 0, path, "docx")],
    }

    let mut reader = Reader::from_str(&doc_xml);
    reader.trim_text(false);
    let mut buf = Vec::new();

    let mut order = 0u32;
    let mut current_page: u32 = 1;
    let mut cur_text = String::new();
    let mut in_t = false;
    let mut pending_heading_level: Option<u8> = None;
    let mut in_p = false;
    let mut para_start_page: u32 = 1;

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                match local_name(e.name().as_ref()) {
                    b"p" => { in_p = true; cur_text.clear(); pending_heading_level = None; para_start_page = current_page; }
                    b"pStyle" => {
                        // Heading style, robust to variants like "Heading1", "Heading 1", case-insensitive
                        if let Some(mut val) = attr_val(&e, b"val") {
                            let lower = val.to_ascii_lowercase();
                            if let Some(rest) = lower.strip_prefix("heading") {
                                let digits: String = rest.chars().filter(|ch| ch.is_ascii_digit()).collect();
                                if let Ok(n) = digits.parse::<u8>() { pending_heading_level = Some(n.max(1)); }
                            }
                        }
                    }
                    b"outlineLvl" => {
                        // Paragraph outline level: 0 => Heading1, 1 => Heading2, ...
                        if let Some(vs) = attr_val(&e, b"val") {
                            if let Ok(n) = vs.parse::<u8>() { pending_heading_level = Some(n.saturating_add(1)); }
                        }
                    }
                    b"t" => { in_t = true; }
                    b"br" => {
                        // page break or line break
                        if let Some(t) = attr_val(&e, b"type") {
                            if t.eq_ignore_ascii_case("page") { current_page = current_page.saturating_add(1); }
                        }
                        cur_text.push('\n');
                    }
                    b"tab" => { cur_text.push('\t'); }
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => {
                match local_name(e.name().as_ref()) {
                    b"br" => {
                        if let Some(t) = attr_val(&e, b"type") {
                            if t.eq_ignore_ascii_case("page") { current_page = current_page.saturating_add(1); }
                        }
                        cur_text.push('\n');
                    }
                    b"tab" => { cur_text.push('\t'); }
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                match local_name(e.name().as_ref()) {
                    b"t" => { in_t = false; }
                    b"p" => {
                        if in_p {
                            let text_s = cur_text.trim();
                            if !text_s.is_empty() {
                                if let Some(level) = pending_heading_level {
                                    let mut b = UnifiedBlock::new(BlockKind::Heading, format!("{}\n", text_s), order, path, "docx");
                                    b.heading_level = Some(level);
                                    b.page_start = Some(para_start_page);
                                    b.page_end = Some(current_page);
                                    blocks.push(b);
                                } else {
                                    let mut b = UnifiedBlock::new(BlockKind::Paragraph, text_s.to_string(), order, path, "docx");
                                    b.page_start = Some(para_start_page);
                                    b.page_end = Some(current_page);
                                    blocks.push(b);
                                }
                                order += 1;
                            }
                            in_p = false; cur_text.clear(); pending_heading_level = None;
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(t)) => {
                if in_t {
                    if let Ok(cow) = t.unescape() { let s: String = cow.into_owned(); cur_text.push_str(&s); }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    if blocks.is_empty() {
        blocks.push(UnifiedBlock::new(BlockKind::Paragraph, String::new(), 0, path, "docx"));
    }
    blocks
}
