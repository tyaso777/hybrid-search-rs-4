use crate::unified_blocks::{BlockKind, UnifiedBlock};
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::collections::HashMap;
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

fn attr_val_q(e: &BytesStart<'_>, qname: &[u8]) -> Option<String> {
    for a in e.attributes().with_checks(false) {
        if let Ok(attr) = a {
            if attr.key.as_ref() == qname {
                return Some(String::from_utf8_lossy(&attr.value).into_owned());
            }
        }
    }
    None
}

/// Read PPTX and convert slides to UnifiedBlocks.
/// - Adds a Heading per slide ("Slide: <title>\n" or fallback "Slide N\n") as level 1
/// - Paragraph blocks for text paragraphs
/// - Tables as TSV wrapped with <table delim="tsv" cell-nl="U+2028">...\n</table>\n and is_table attr
pub fn read_pptx_to_blocks(path: &str) -> Vec<UnifiedBlock> {
    let mut blocks: Vec<UnifiedBlock> = Vec::new();
    let file = match File::open(path) { Ok(f) => f, Err(_) => return vec![UnifiedBlock::new(BlockKind::Paragraph, "(error) failed to open PPTX", 0, path, "pptx")] };
    let mut zip = match zip::ZipArchive::new(file) { Ok(z) => z, Err(_) => return vec![UnifiedBlock::new(BlockKind::Paragraph, "(error) not a valid .pptx (zip) file", 0, path, "pptx")] };

    // 1) Read relationships for presentation.xml (map rId -> target)
    let mut rels_map: HashMap<String, String> = HashMap::new();
    if let Ok(mut rels) = zip.by_name("ppt/_rels/presentation.xml.rels") {
        let mut rels_xml = String::new(); let _ = rels.read_to_string(&mut rels_xml);
        let mut r = Reader::from_str(&rels_xml); r.trim_text(false);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match r.read_event_into(&mut buf) {
                Ok(Event::Empty(e)) | Ok(Event::Start(e)) => {
                    if local_name(e.name().as_ref()) == b"Relationship" {
                        let id = attr_val(&e, b"Id");
                        let target = attr_val(&e, b"Target");
                        if let (Some(id), Some(mut t)) = (id, target) {
                            if !t.starts_with("ppt/") { t = format!("ppt/{}", t); }
                            rels_map.insert(id, t);
                        }
                    }
                }
                Ok(Event::Eof) => break,
                Err(_) => break,
                _ => {}
            }
        }
    }

    // 2) Read presentation.xml to get slide order (p:sldId r:id)
    let mut pres_xml = String::new();
    if let Ok(mut pres) = zip.by_name("ppt/presentation.xml") { let _ = pres.read_to_string(&mut pres_xml); } else { return blocks; }
    let mut reader = Reader::from_str(&pres_xml);
    reader.trim_text(false);
    let mut buf = Vec::new();
    let mut slide_targets: Vec<String> = Vec::new();
    let mut slide_cy_opt: Option<i64> = None;
    loop {
        buf.clear();
        match reader.read_event_into(&mut buf) {
            Ok(Event::Empty(e)) | Ok(Event::Start(e)) => {
                if local_name(e.name().as_ref()) == b"sldId" {
                    // prefer r:id (relationship id); ignore numeric id
                    if let Some(rid) = attr_val_q(&e, b"r:id") {
                        if let Some(t) = rels_map.get(&rid) { slide_targets.push(t.clone()); }
                    }
                } else if local_name(e.name().as_ref()) == b"sldSz" {
                    if slide_cy_opt.is_none() {
                        if let Some(v) = attr_val(&e, b"cy") { if let Ok(n) = v.parse::<i64>() { slide_cy_opt = Some(n); } }
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    if slide_targets.is_empty() { return blocks; }

    let mut order = 0u32;
    for (i, tgt) in slide_targets.iter().enumerate() {
        let slide_num = (i as u32) + 1;
        // Insert provisional heading; update later if title found
        let mut h = UnifiedBlock::new(BlockKind::Heading, format!("Slide {}\n", slide_num), order, path, "pptx");
        h.heading_level = Some(1);
        h.page_start = Some(slide_num);
        h.page_end = Some(slide_num);
        let heading_idx = blocks.len();
        blocks.push(h);
        order += 1;

        // Read slide XML
        let mut slide_xml = String::new();
        match zip.by_name(&tgt) {
            Ok(mut f) => { let _ = f.read_to_string(&mut slide_xml); }
            Err(_) => continue,
        }
        let mut r = Reader::from_str(&slide_xml);
        r.trim_text(false);
        let mut buf = Vec::new();

        // State
        let mut in_sp = false; // shape
        let mut is_title_shape = false;
        let mut in_tx = false; // inside a:txBody
        let mut in_p = false;  // a:p
        let mut in_t = false;  // a:t
        let mut cur_text = String::new();
        let mut para_texts: Vec<String> = Vec::new();
        let mut slide_title: Option<String> = None;

        let mut in_tbl = false; let mut in_tr = false; let mut in_tc = false;
        let mut cell_text = String::new(); let mut row_cells: Vec<String> = Vec::new(); let mut table_text = String::new();

        // Position state (EMU) — kept but unused after revert
        let mut _in_sp_xfrm = false; let mut _cur_x: i64 = 0; let mut _cur_y: i64 = 0; let mut _cur_cy: i64 = 0;
        let mut _in_gf = false; let mut _in_gf_xfrm = false; let mut _tbl_x: i64 = 0; let mut _tbl_y: i64 = 0; let mut _tbl_cy: i64 = 0;

        #[derive(Debug)]
        struct SlideItem { x: i64, y: i64, cy: i64, blocks: Vec<UnifiedBlock> }
        let mut items: Vec<SlideItem> = Vec::new();

        loop {
            buf.clear();
            match r.read_event_into(&mut buf) {
                Ok(Event::Start(e)) => {
                    match local_name(e.name().as_ref()) {
                        b"sp" => { in_sp = true; is_title_shape = false; para_texts.clear(); _cur_x=0; _cur_y=0; _cur_cy=0; }
                        b"ph" => {
                            // title/ctrTitle/subTitle
                            if let Some(t) = attr_val(&e, b"type") {
                                let lt = t.to_ascii_lowercase();
                                if lt == "title" || lt == "ctrtitle" || lt == "subtitle" { is_title_shape = true; }
                            }
                        }
                        b"txBody" => { in_tx = true; }
                        b"p" => { if in_tx { in_p = true; cur_text.clear(); } }
                        b"t" => { if in_tx && in_p { in_t = true; } }
                        b"spPr" => { /* container for xfrm */ }
                        b"xfrm" => { if in_sp { _in_sp_xfrm = true; } else if _in_gf { _in_gf_xfrm = true; } }
                        b"off" => {
                            if _in_sp_xfrm { if let Some(v) = attr_val(&e, b"x") { if let Ok(n)=v.parse::<i64>(){ _cur_x=n; } } if let Some(v) = attr_val(&e, b"y") { if let Ok(n)=v.parse::<i64>(){ _cur_y=n; } } }
                            if _in_gf_xfrm { if let Some(v) = attr_val(&e, b"x") { if let Ok(n)=v.parse::<i64>(){ _tbl_x=n; } } if let Some(v) = attr_val(&e, b"y") { if let Ok(n)=v.parse::<i64>(){ _tbl_y=n; } } }
                        }
                        b"ext" => {
                            if _in_sp_xfrm { if let Some(v) = attr_val(&e, b"cy") { if let Ok(n)=v.parse::<i64>(){ _cur_cy=n; } } }
                            if _in_gf_xfrm { if let Some(v) = attr_val(&e, b"cy") { if let Ok(n)=v.parse::<i64>(){ _tbl_cy=n; } } }
                        }
                        b"graphicFrame" => { _in_gf = true; _tbl_x=0; _tbl_y=0; _tbl_cy=0; }
                        b"tbl" => { in_tbl = true; table_text.clear(); row_cells.clear(); }
                        b"tr" => { if in_tbl { in_tr = true; row_cells.clear(); } }
                        b"tc" => { if in_tr { in_tc = true; cell_text.clear(); } }
                        _ => {}
                    }
                }
                Ok(Event::Empty(e)) => {
                    match local_name(e.name().as_ref()) {
                        b"ph" => {
                            if let Some(t) = attr_val(&e, b"type") {
                                let lt = t.to_ascii_lowercase();
                                if lt == "title" || lt == "ctrtitle" || lt == "subtitle" { is_title_shape = true; }
                            }
                        }
                        b"t" => { if in_tx && in_p { /* xml:space ignored */ } }
                        b"xfrm" => { if in_sp { _in_sp_xfrm = true; } else if _in_gf { _in_gf_xfrm = true; } }
                        b"off" => {
                            if _in_sp_xfrm { if let Some(v) = attr_val(&e, b"x") { if let Ok(n)=v.parse::<i64>(){ _cur_x=n; } } if let Some(v) = attr_val(&e, b"y") { if let Ok(n)=v.parse::<i64>(){ _cur_y=n; } } }
                            if _in_gf_xfrm { if let Some(v) = attr_val(&e, b"x") { if let Ok(n)=v.parse::<i64>(){ _tbl_x=n; } } if let Some(v) = attr_val(&e, b"y") { if let Ok(n)=v.parse::<i64>(){ _tbl_y=n; } } }
                        }
                        b"ext" => {
                            if _in_sp_xfrm { if let Some(v) = attr_val(&e, b"cy") { if let Ok(n)=v.parse::<i64>(){ _cur_cy=n; } } }
                            if _in_gf_xfrm { if let Some(v) = attr_val(&e, b"cy") { if let Ok(n)=v.parse::<i64>(){ _tbl_cy=n; } } }
                        }
                        b"tbl" => { in_tbl = true; }
                        b"tr" => { if in_tbl { in_tr = true; } }
                        b"tc" => { if in_tr { in_tc = true; } }
                        _ => {}
                    }
                }
                Ok(Event::End(e)) => {
                    match local_name(e.name().as_ref()) {
                        b"t" => { in_t = false; }
                        b"p" => {
                            if in_tx && in_p {
                                let text = cur_text.trim().to_string();
                                if !text.is_empty() { para_texts.push(text); }
                                cur_text.clear(); in_p = false;
                            }
                        }
                        b"txBody" => {
                            if in_sp && in_tx {
                                // flush shape text
                                if !para_texts.is_empty() {
                                    if is_title_shape && slide_title.is_none() {
                                        // use first non-empty as title
                                        slide_title = para_texts.iter().find(|s| !s.trim().is_empty()).cloned();
                                    } else {
                                        for ptxt in para_texts.drain(..) {
                                            let mut b = UnifiedBlock::new(BlockKind::Paragraph, ptxt, order, path, "pptx");
                                            b.page_start = Some(slide_num); b.page_end = Some(slide_num);
                                            blocks.push(b); order += 1;
                                        }
                                    }
                                }
                                para_texts.clear(); in_tx = false;
                            }
                        }
                        b"sp" => { in_sp = false; is_title_shape = false; _in_sp_xfrm = false; }
                        b"xfrm" => { _in_sp_xfrm = false; _in_gf_xfrm = false; }
                        b"graphicFrame" => { _in_gf = false; _in_gf_xfrm = false; }
                        b"tc" => {
                            if in_tc {
                                in_tc = false;
                                let cell = cell_text.replace("\r\n", "\n").replace('\r', "\n").replace('\n', "\u{2028}");
                                row_cells.push(cell); cell_text.clear();
                            }
                        }
                        b"tr" => {
                            if in_tr {
                                in_tr = false;
                                let line = row_cells.join("\t");
                                table_text.push_str(&line); table_text.push('\n');
                                row_cells.clear();
                            }
                        }
                        b"tbl" => {
                            if in_tbl {
                                in_tbl = false;
                                if !table_text.is_empty() {
                                    let content = table_text.trim_end_matches('\n');
                                    let mut wrapped = String::new();
                                    let need_leading_nl = blocks.last().map_or(false, |prev| !prev.text.ends_with('\n'));
                                    if need_leading_nl { wrapped.push('\n'); }
                                    wrapped.push_str(&format!("<table delim=\"tsv\" cell-nl=\"U+2028\">\n{}\n</table>\n", content));
                                    let mut b = UnifiedBlock::new(BlockKind::Paragraph, wrapped, order, path, "pptx");
                                    b.page_start = Some(slide_num); b.page_end = Some(slide_num);
                                    b.attrs.insert("is_table".to_string(), "true".to_string());
                                    b.attrs.insert("table_cell_nl".to_string(), "U+2028".to_string());
                                    blocks.push(b); order += 1;
                                }
                                table_text.clear();
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::Text(t)) => {
                    if in_t { if let Ok(cow) = t.unescape() { cur_text.push_str(&cow); } }
                    if in_tc && !in_t { /* no-op */ }
                }
                Ok(Event::Eof) => break,
                Err(_) => break,
                _ => {}
            }
        }

        // Update heading with title if available
        if let Some(title) = slide_title.take() {
            if let Some(hb) = blocks.get_mut(heading_idx) {
                hb.text = format!("Slide: {}\n", title);
            }
        }

        // Now sort slide items by y (top->bottom), then x (left->right) and append to blocks
        items.sort_by(|a, b| {
            let oy = a.y.cmp(&b.y);
            if oy == std::cmp::Ordering::Equal { a.x.cmp(&b.x) } else { oy }
        });

        let slide_h = slide_cy_opt.unwrap_or(6_858_000); // fallback ~7e6 EMU
        let epsilon: i64 = std::cmp::max((slide_h as f64 * 0.01) as i64, 50_000);

        let mut prev_item: Option<(i64, i64, i64, usize)> = None; // (x,y,cy,last_global_idx)
        for item in items.into_iter() {
            // If moving to a lower row, inject a newline at the end of the previous item's last block (if missing)
            if let Some((_px, py, pcy, last_idx)) = prev_item {
                let same_row = {
                    let a_top = py; let a_bot = py + pcy.max(1);
                    let b_top = item.y; let b_bot = item.y + item.cy.max(1);
                    let overlap = std::cmp::min(a_bot, b_bot) - std::cmp::max(a_top, b_top);
                    let min_h = std::cmp::min(pcy.max(1), item.cy.max(1));
                    let overlap_ratio_ok = overlap > 0 && (overlap as f64) / (min_h as f64) >= 0.3;
                    let mid_a = py + pcy / 2; let mid_b = item.y + item.cy / 2;
                    let epsilon_ok = (mid_a - mid_b).abs() <= epsilon;
                    overlap_ratio_ok || epsilon_ok
                };
                if !same_row {
                    if let Some(prev_blk) = blocks.get_mut(last_idx) {
                        if !prev_blk.text.ends_with('\n') { prev_blk.text.push('\n'); }
                    }
                }
            }

            let mut last_global_idx: Option<usize> = None;
            for b in item.blocks.into_iter() {
                let idx = blocks.len();
                blocks.push(b);
                last_global_idx = Some(idx);
            }
            if let Some(idx) = last_global_idx { prev_item = Some((item.x, item.y, item.cy, idx)); }
        }
    }

    if blocks.is_empty() { blocks.push(UnifiedBlock::new(BlockKind::Paragraph, String::new(), 0, path, "pptx")); }
    blocks
}
