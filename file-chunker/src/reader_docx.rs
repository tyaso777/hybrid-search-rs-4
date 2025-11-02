use crate::unified_blocks::{UnifiedBlock, BlockKind};
use quick_xml::events::{BytesStart, Event};
use quick_xml::Reader;
use std::collections::{HashMap, HashSet};
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

/// Parse word/styles.xml and build a map of paragraph styleId -> heading level (1-based)
/// by resolving w:pPr/w:outlineLvl (0-based) and following basedOn chains.
fn parse_styles_outline_levels<R: Read + std::io::Seek>(zip: &mut zip::ZipArchive<R>) -> HashMap<String, u8> {
    let mut xml = String::new();
    if let Ok(mut f) = zip.by_name("word/styles.xml") {
        let _ = f.read_to_string(&mut xml);
    } else {
        return HashMap::new();
    }

    #[derive(Clone, Debug, Default)]
    struct StyleTmp { based_on: Option<String>, outline_level: Option<u8> }
    let mut map: HashMap<String, StyleTmp> = HashMap::new();

    let mut reader = Reader::from_str(&xml);
    reader.trim_text(false);
    let mut buf = Vec::new();
    let mut cur_id: Option<String> = None;
    let mut cur_is_para: bool = false;

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                match local_name(e.name().as_ref()) {
                    b"style" => {
                        let mut is_para = false;
                        let mut sid: Option<String> = None;
                        for a in e.attributes().with_checks(false) {
                            if let Ok(attr) = a {
                                let k = local_name(attr.key.as_ref());
                                if k == b"type" {
                                    if String::from_utf8_lossy(&attr.value).eq_ignore_ascii_case("paragraph") { is_para = true; }
                                } else if k == b"styleId" { sid = Some(String::from_utf8_lossy(&attr.value).into_owned()); }
                            }
                        }
                        cur_is_para = is_para;
                        cur_id = if is_para { sid } else { None };
                        if let Some(ref id) = cur_id { map.entry(id.clone()).or_default(); }
                    }
                    b"basedOn" => {
                        if cur_is_para { if let Some(ref id) = cur_id { if let Some(v) = attr_val(&e, b"val") { map.entry(id.clone()).or_default().based_on = Some(v); } } }
                    }
                    b"outlineLvl" => {
                        if cur_is_para { if let Some(ref id) = cur_id { if let Some(vs) = attr_val(&e, b"val") { if let Ok(n) = vs.parse::<u8>() { map.entry(id.clone()).or_default().outline_level = Some(n.saturating_add(1)); } } } }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => {
                match local_name(e.name().as_ref()) {
                    b"basedOn" => {
                        if cur_is_para { if let Some(ref id) = cur_id { if let Some(v) = attr_val(&e, b"val") { map.entry(id.clone()).or_default().based_on = Some(v); } } }
                    }
                    b"outlineLvl" => {
                        if cur_is_para { if let Some(ref id) = cur_id { if let Some(vs) = attr_val(&e, b"val") { if let Ok(n) = vs.parse::<u8>() { map.entry(id.clone()).or_default().outline_level = Some(n.saturating_add(1)); } } } }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                if local_name(e.name().as_ref()) == b"style" { cur_id = None; cur_is_para = false; }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    // Resolve based_on chains
    let mut resolved: HashMap<String, u8> = HashMap::new();
    let mut visiting: HashSet<String> = HashSet::new();

    fn resolve(id: &str, map: &HashMap<String, StyleTmp>, resolved: &mut HashMap<String, u8>, visiting: &mut HashSet<String>) -> Option<u8> {
        if let Some(&lvl) = resolved.get(id) { return Some(lvl); }
        if !visiting.insert(id.to_string()) { return None; }
        let out = if let Some(st) = map.get(id) {
            if let Some(l) = st.outline_level { Some(l) } else if let Some(ref parent) = st.based_on { resolve(parent, map, resolved, visiting) } else { None }
        } else { None };
        visiting.remove(id);
        if let Some(l) = out { resolved.insert(id.to_string(), l); }
        out
    }

    for id in map.keys() { let _ = resolve(id, &map, &mut resolved, &mut visiting); }
    resolved
}

/// Parse styles.xml and return paragraph styleId -> (numId, ilvl) default numbering mapping.
fn parse_styles_numpr_map<R: Read + std::io::Seek>(zip: &mut zip::ZipArchive<R>) -> HashMap<String, (Option<u32>, Option<u8>)> {
    let mut xml = String::new();
    if let Ok(mut f) = zip.by_name("word/styles.xml") {
        let _ = f.read_to_string(&mut xml);
    } else { return HashMap::new(); }

    let mut reader = Reader::from_str(&xml);
    reader.trim_text(false);
    let mut buf = Vec::new();
    let mut cur_id: Option<String> = None; let mut cur_is_para = false; let mut in_numpr = false;
    let mut out: HashMap<String, (Option<u32>, Option<u8>)> = HashMap::new();

    loop { buf.clear(); match reader.read_event_into(&mut buf) {
        Ok(Event::Start(e)) => {
            match local_name(e.name().as_ref()) {
                b"style" => {
                    let mut is_para=false; let mut sid: Option<String>=None;
                    for a in e.attributes().with_checks(false) { if let Ok(attr)=a { let k=local_name(attr.key.as_ref()); if k==b"type" { if String::from_utf8_lossy(&attr.value).eq_ignore_ascii_case("paragraph") { is_para=true; } } else if k==b"styleId" { sid=Some(String::from_utf8_lossy(&attr.value).into_owned()); } } }
                    cur_is_para = is_para; cur_id = if is_para { sid } else { None };
                }
                b"numPr" => { if cur_is_para { in_numpr = true; } }
                b"numId" => { if cur_is_para && in_numpr { if let Some(ref id)=cur_id { if let Some(v)=attr_val(&e, b"val") { if let Ok(n)=v.parse::<u32>() { out.entry(id.clone()).or_insert((None,None)).0 = Some(n); } } } } }
                b"ilvl" => { if cur_is_para && in_numpr { if let Some(ref id)=cur_id { if let Some(v)=attr_val(&e, b"val") { if let Ok(n)=v.parse::<u8>() { out.entry(id.clone()).or_insert((None,None)).1 = Some(n); } } } } }
                _ => {}
            }
        }
        Ok(Event::Empty(e)) => {
            match local_name(e.name().as_ref()) {
                b"style" => {
                    let mut is_para=false; let mut sid: Option<String>=None;
                    for a in e.attributes().with_checks(false) { if let Ok(attr)=a { let k=local_name(attr.key.as_ref()); if k==b"type" { if String::from_utf8_lossy(&attr.value).eq_ignore_ascii_case("paragraph") { is_para=true; } } else if k==b"styleId" { sid=Some(String::from_utf8_lossy(&attr.value).into_owned()); } } }
                    cur_is_para = is_para; cur_id = None; if is_para { if let Some(id)=sid { out.entry(id).or_insert((None,None)); } }
                }
                b"numPr" => { if cur_is_para { in_numpr = true; } }
                b"numId" => { if cur_is_para && in_numpr { if let Some(ref id)=cur_id { if let Some(v)=attr_val(&e, b"val") { if let Ok(n)=v.parse::<u32>() { out.entry(id.clone()).or_insert((None,None)).0 = Some(n); } } } } }
                b"ilvl" => { if cur_is_para && in_numpr { if let Some(ref id)=cur_id { if let Some(v)=attr_val(&e, b"val") { if let Ok(n)=v.parse::<u8>() { out.entry(id.clone()).or_insert((None,None)).1 = Some(n); } } } } }
                _ => {}
            }
        }
        Ok(Event::End(e)) => {
            match local_name(e.name().as_ref()) {
                b"numPr" => { in_numpr = false; }
                b"style" => { cur_id=None; cur_is_para=false; in_numpr=false; }
                _ => {}
            }
        }
        Ok(Event::Eof) => break,
        Err(_) => break,
        _ => {}
    }}
    out
}

// ---------------- Numbering (lists/headings numbering) ----------------

#[derive(Clone, Debug, Default)]
struct LvlDef { text: Option<String>, fmt: Option<String>, suff: Option<String>, start: i32 }

#[derive(Default)]
struct NumberingCfg {
    abs: HashMap<u32, HashMap<u8, LvlDef>>,                 // abstractNumId -> lvl -> def
    nums: HashMap<u32, (u32, HashMap<u8, i32>)>,            // numId -> (abstractNumId, overrides start per lvl)
    style_abs: HashMap<String, (u32, u8)>,                  // styleId -> (abstractNumId, ilvl)
}

#[derive(Default)]
struct NumberingState { counters: HashMap<u32, Vec<i32>> }   // numId -> counters per level

#[derive(Default)]
struct Numbering { cfg: NumberingCfg, st: NumberingState }

fn parse_numbering<R: Read + std::io::Seek>(zip: &mut zip::ZipArchive<R>) -> NumberingCfg {
    let mut xml = String::new();
    if let Ok(mut f) = zip.by_name("word/numbering.xml") {
        let _ = f.read_to_string(&mut xml);
    } else { return NumberingCfg::default(); }

    let mut cfg = NumberingCfg::default();
    let mut reader = Reader::from_str(&xml);
    reader.trim_text(false);
    let mut buf = Vec::new();

    let mut in_abs: Option<u32> = None;
    let mut cur_lvl: Option<u8> = None;
    let mut in_num: Option<u32> = None;
    let mut cur_override_lvl: Option<u8> = None;

    loop { buf.clear(); match reader.read_event_into(&mut buf) {
        Ok(Event::Start(e)) => {
            match local_name(e.name().as_ref()) {
                b"abstractNum" => {
                    in_abs = attr_val(&e, b"abstractNumId").and_then(|s| s.parse::<u32>().ok());
                    if let Some(id) = in_abs { cfg.abs.entry(id).or_default(); }
                }
                b"lvl" => {
                    cur_lvl = attr_val(&e, b"ilvl").and_then(|s| s.parse::<u8>().ok());
                    if let (Some(abs), Some(lv)) = (in_abs, cur_lvl) { cfg.abs.entry(abs).or_default().entry(lv).or_insert_with(LvlDef::default); }
                }
                b"lvlText" => {
                    if let (Some(abs), Some(lv)) = (in_abs, cur_lvl) {
                        if let Some(v) = attr_val(&e, b"val") { cfg.abs.entry(abs).or_default().entry(lv).or_default().text = Some(v); }
                    }
                }
                b"pStyle" => {
                    if let (Some(abs), Some(lv)) = (in_abs, cur_lvl) {
                        if let Some(sid) = attr_val(&e, b"val") {
                            cfg.style_abs
                                .entry(sid)
                                .and_modify(|(a, existing_lv)| { if lv < *existing_lv { *a = abs; *existing_lv = lv; } })
                                .or_insert((abs, lv));
                        }
                    }
                }
                b"numFmt" => {
                    if let (Some(abs), Some(lv)) = (in_abs, cur_lvl) {
                        if let Some(v) = attr_val(&e, b"val") { cfg.abs.entry(abs).or_default().entry(lv).or_default().fmt = Some(v.to_ascii_lowercase()); }
                    }
                }
                b"suff" => {
                    if let (Some(abs), Some(lv)) = (in_abs, cur_lvl) {
                        if let Some(v) = attr_val(&e, b"val") { cfg.abs.entry(abs).or_default().entry(lv).or_default().suff = Some(v.to_ascii_lowercase()); }
                    }
                }
                b"start" => {
                    if let (Some(abs), Some(lv)) = (in_abs, cur_lvl) {
                        if let Some(v) = attr_val(&e, b"val") { if let Ok(n) = v.parse::<i32>() { cfg.abs.entry(abs).or_default().entry(lv).or_default().start = n; } }
                    }
                }
                b"num" => {
                    in_num = attr_val(&e, b"numId").and_then(|s| s.parse::<u32>().ok());
                    if let Some(id) = in_num { cfg.nums.entry(id).or_insert((0, HashMap::new())); }
                }
                b"abstractNumId" => {
                    if let Some(num) = in_num { if let Some(v) = attr_val(&e, b"val") { if let Ok(an) = v.parse::<u32>() { cfg.nums.entry(num).or_insert((0, HashMap::new())).0 = an; } } }
                }
                b"lvlOverride" => {
                    cur_override_lvl = attr_val(&e, b"ilvl").and_then(|s| s.parse::<u8>().ok());
                }
                b"startOverride" => {
                    if let (Some(num), Some(lv)) = (in_num, cur_override_lvl) {
                        if let Some(v) = attr_val(&e, b"val") { if let Ok(n) = v.parse::<i32>() { cfg.nums.entry(num).or_insert((0, HashMap::new())).1.insert(lv, n); } }
                    }
                }
                _ => {}
            }
        }
        Ok(Event::Empty(e)) => {
            match local_name(e.name().as_ref()) {
                b"abstractNum" => { in_abs = attr_val(&e, b"abstractNumId").and_then(|s| s.parse::<u32>().ok()); if let Some(id) = in_abs { cfg.abs.entry(id).or_default(); } in_abs=None; }
                b"lvl" => { cur_lvl = attr_val(&e, b"ilvl").and_then(|s| s.parse::<u8>().ok()); if let (Some(abs), Some(lv)) = (in_abs, cur_lvl) { cfg.abs.entry(abs).or_default().entry(lv).or_insert_with(LvlDef::default); } cur_lvl=None; }
                b"lvlText" => { if let (Some(abs), Some(lv)) = (in_abs, cur_lvl) { if let Some(v) = attr_val(&e, b"val") { cfg.abs.entry(abs).or_default().entry(lv).or_default().text = Some(v); } } }
                b"pStyle" => {
                    if let (Some(abs), Some(lv)) = (in_abs, cur_lvl) {
                        if let Some(sid) = attr_val(&e, b"val") {
                            cfg.style_abs
                                .entry(sid)
                                .and_modify(|(a, existing_lv)| { if lv < *existing_lv { *a = abs; *existing_lv = lv; } })
                                .or_insert((abs, lv));
                        }
                    }
                }
                b"numFmt" => { if let (Some(abs), Some(lv)) = (in_abs, cur_lvl) { if let Some(v) = attr_val(&e, b"val") { cfg.abs.entry(abs).or_default().entry(lv).or_default().fmt = Some(v.to_ascii_lowercase()); } } }
                b"suff" => { if let (Some(abs), Some(lv)) = (in_abs, cur_lvl) { if let Some(v) = attr_val(&e, b"val") { cfg.abs.entry(abs).or_default().entry(lv).or_default().suff = Some(v.to_ascii_lowercase()); } } }
                b"start" => { if let (Some(abs), Some(lv)) = (in_abs, cur_lvl) { if let Some(v) = attr_val(&e, b"val") { if let Ok(n) = v.parse::<i32>() { cfg.abs.entry(abs).or_default().entry(lv).or_default().start = n; } } } }
                b"num" => { if let Some(num) = attr_val(&e, b"numId").and_then(|s| s.parse::<u32>().ok()) { cfg.nums.entry(num).or_insert((0, HashMap::new())); } }
                b"abstractNumId" => { if let Some(num) = in_num { if let Some(v) = attr_val(&e, b"val") { if let Ok(an) = v.parse::<u32>() { cfg.nums.entry(num).or_insert((0, HashMap::new())).0 = an; } } } }
                b"lvlOverride" => { cur_override_lvl = attr_val(&e, b"ilvl").and_then(|s| s.parse::<u8>().ok()); cur_override_lvl=None; }
                b"startOverride" => { if let (Some(num), Some(lv)) = (in_num, cur_override_lvl) { if let Some(v) = attr_val(&e, b"val") { if let Ok(n) = v.parse::<i32>() { cfg.nums.entry(num).or_insert((0, HashMap::new())).1.insert(lv, n); } } } }
                _ => {}
            }
        }
        Ok(Event::End(e)) => {
            match local_name(e.name().as_ref()) {
                b"abstractNum" => { in_abs=None; }
                b"lvl" => { cur_lvl=None; }
                b"num" => { in_num=None; }
                _ => {}
            }
        }
        Ok(Event::Eof) => break,
        Err(_) => break,
        _ => {}
    }}
    cfg
}

impl Numbering {
    fn from_cfg(cfg: NumberingCfg) -> Self { Self { cfg, st: NumberingState::default() } }

    fn format_prefix(&mut self, num_id: u32, ilvl: u8) -> String {
        let (abs_id, overrides) = match self.cfg.nums.get(&num_id) { Some(x) => x, None => return String::new() };
        let defs = match self.cfg.abs.get(abs_id) { Some(d) => d, None => return String::new() };
        let cur_def = defs.get(&ilvl).cloned().unwrap_or_default();
        let start_for = |lv: u8| -> i32 {
            if let Some(v) = overrides.get(&lv) { *v } else {
                let d = defs.get(&lv).cloned().unwrap_or_default();
                if d.start != 0 { d.start } else { 1 }
            }
        };
        let start_val = start_for(ilvl);

        let counters = self.st.counters.entry(num_id).or_insert_with(Vec::new);
        if counters.len() <= ilvl as usize { counters.resize((ilvl as usize)+1, 0); }
        // Reset deeper levels
        for j in (ilvl as usize + 1)..counters.len() { counters[j] = 0; }
        // Initialize only ancestor levels (exclude current) so %1..%(ilvl-1) never render as 0
        if ilvl > 0 {
            for k in 0..(ilvl as usize) {
                if counters[k] == 0 { counters[k] = start_for(k as u8); }
            }
        }
        // Increment current level: initialize on first occurrence, then increment on subsequent ones
        if counters[ilvl as usize] == 0 {
            counters[ilvl as usize] = start_val;
        } else {
            counters[ilvl as usize] += 1;
        }

        // Build text by replacing %1..%9 in lvlText (Unicode-safe)
        let tmpl = cur_def.text.unwrap_or_else(|| "%1".to_string());
        let chars: Vec<char> = tmpl.chars().collect();
        let mut out = String::new();
        let mut i = 0usize;
        while i < chars.len() {
            if chars[i] == '%' && i + 1 < chars.len() && chars[i+1].is_ascii_digit() {
                let mut j = i + 1;
                let mut num: usize = 0;
                while j < chars.len() && chars[j].is_ascii_digit() {
                    num = num * 10 + (chars[j] as u8 - b'0') as usize;
                    j += 1;
                }
                if (1..=9).contains(&num) {
                    let lv_idx = (num as u8).saturating_sub(1);
                    let val = if (lv_idx as usize) < counters.len() { counters[lv_idx as usize] } else { 0 };
                    let fmt = defs.get(&lv_idx).and_then(|d| d.fmt.clone());
                    out.push_str(&format_number(val, fmt.as_deref()));
                    i = j; continue;
                }
            }
            out.push(chars[i]);
            i += 1;
        }
        // suffix from current level definition
        match cur_def.suff.as_deref() {
            Some("tab") => out.push('\t'),
            Some("space") => out.push(' '),
            _ => {}
        }
        out
    }

    /// When a paragraph has a style but no explicit numPr, try to infer (numId, ilvl)
    /// from numbering.xml's pStyle binding on abstractNum/lvl and the available nums
    fn style_default_num(&self, style_id: &str) -> Option<(u32, u8)> {
        let (abs_id, ilvl) = *self.cfg.style_abs.get(style_id)?;
        // Find a numId that references this abstract num. Pick the smallest for stability.
        let mut best: Option<u32> = None;
        for (num_id, (aid, _)) in &self.cfg.nums {
            if *aid == abs_id {
                match best { Some(b) if *num_id >= b => {}, _ => best = Some(*num_id) }
            }
        }
        best.map(|n| (n, ilvl))
    }
}

fn format_number(val: i32, fmt: Option<&str>) -> String {
    let n = if val <= 0 { 0 } else { val } as i32;
    let f = fmt.unwrap_or("decimal").to_ascii_lowercase();
    match f.as_str() {
        // Arabic numerals
        "decimal" | "decimalzero" | "decimalzero1" => format!("{}", n),
        "decimalfullwidth" => to_fullwidth_decimal(n as i64),
        // Roman
        "upperroman" => to_roman(n as i64, true),
        "lowerroman" => to_roman(n as i64, false),
        // Latin letters
        "upperletter" => to_alpha(n as i64, true),
        "lowerletter" => to_alpha(n as i64, false),
        // Japanese kana families
        // Wordの挙動に合わせ、aiueoFullWidth はカタカナ相当として扱う
        "aiueofullwidth" => to_katakana_aiueo(n as usize),
        "aiueo" => to_hiragana_aiueo(n as usize),
        x if x.contains("katakana") => to_katakana_aiueo(n as usize),
        x if x.contains("hiragana") => to_hiragana_aiueo(n as usize),
        _ => format!("{}", n),
    }
}

fn to_fullwidth_decimal(n: i64) -> String {
    let s = format!("{}", n);
    s.chars().map(|c| match c { '0'..='9' => char::from_u32('０' as u32 + (c as u32 - '0' as u32)).unwrap(), _ => c }).collect()
}

fn to_alpha(mut n: i64, upper: bool) -> String {
    if n <= 0 { return String::new(); }
    let base = if upper { 'A' } else { 'a' } as u32; let mut s = String::new();
    while n > 0 { n -= 1; let ch = (base + (n % 26) as u32) as u8 as char; s.insert(0, ch); n /= 26; }
    s
}

fn to_roman(mut n: i64, upper: bool) -> String {
    if n <= 0 { return String::new(); }
    let vals = [1000,900,500,400,100,90,50,40,10,9,5,4,1];
    let syms = ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"]; let mut out=String::new();
    for (v, s) in vals.iter().zip(syms.iter()) { while n >= *v { out.push_str(s); n -= *v; } }
    if !upper { out = out.to_lowercase(); }
    out
}

fn to_katakana_aiueo(n: usize) -> String {
    const K: &[&str] = &["ア","イ","ウ","エ","オ","カ","キ","ク","ケ","コ","サ","シ","ス","セ","ソ","タ","チ","ツ","テ","ト","ナ","ニ","ヌ","ネ","ノ","ハ","ヒ","フ","ヘ","ホ","マ","ミ","ム","メ","モ","ヤ","ユ","ヨ","ラ","リ","ル","レ","ロ","ワ","ヲ","ン"];
    if n==0 { return String::new(); }
    if n <= K.len() { K[n-1].to_string() } else { format!("{}", n) }
}

fn to_hiragana_aiueo(n: usize) -> String {
    const H: &[&str] = &["あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ","た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ","ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん"];
    if n==0 { return String::new(); }
    if n <= H.len() { H[n-1].to_string() } else { format!("{}", n) }
}

/// Minimal DOCX reader: opens the zip, parses word/document.xml, extracts paragraph text and heading level.
pub fn read_docx_to_blocks(path: &str) -> Vec<UnifiedBlock> {
    let mut blocks: Vec<UnifiedBlock> = Vec::new();
    let file = match File::open(path) { Ok(f) => f, Err(_) => return vec![UnifiedBlock::new(BlockKind::Paragraph, "(error) failed to open DOCX", 0, path, "docx")] };
    let mut zip = match zip::ZipArchive::new(file) { Ok(z) => z, Err(_) => return vec![UnifiedBlock::new(BlockKind::Paragraph, "(error) not a valid .docx (zip) file", 0, path, "docx")] };
    // Parse styles.xml to resolve styleId -> heading level (1-based) via outlineLvl (0-based in XML)
    let style_levels = parse_styles_outline_levels(&mut zip);
    // Parse styles.xml to resolve styleId -> default numbering (numId/ilvl) when paragraph lacks numPr
    let style_numpr = parse_styles_numpr_map(&mut zip);
    // Parse numbering.xml for list/heading numbering
    let numbering_cfg = parse_numbering(&mut zip);
    let mut numbering = Numbering::from_cfg(numbering_cfg);
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
    let mut current_style_id: Option<String> = None;
    let mut in_numpr = false; let mut pending_num_id: Option<u32> = None; let mut pending_ilvl: Option<u8> = None;
    let mut in_p = false;
    let mut para_start_page: u32 = 1;

    // Table extraction state: join cells with '\t' and rows with '\n'
    let mut in_tbl = false;
    let mut in_tr = false;
    let mut in_tc = false;
    let mut cell_text = String::new();
    let mut row_cells: Vec<String> = Vec::new();
    let mut table_text = String::new();
    let mut table_start_page: u32 = 1;

    loop {
        buf.clear();
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                match local_name(e.name().as_ref()) {
                    b"p" => {
                        if !in_tc { // paragraphs inside table cells are aggregated into the cell text
                            in_p = true; cur_text.clear(); pending_heading_level = None; para_start_page = current_page; current_style_id = None;
                        }
                    }
                    b"tbl" => { in_tbl = true; table_text.clear(); table_start_page = current_page; }
                    b"tr" => { if in_tbl { in_tr = true; row_cells.clear(); } }
                    b"tc" => { if in_tr { in_tc = true; cell_text.clear(); } }
                    b"numPr" => { in_numpr = true; pending_num_id=None; pending_ilvl=None; }
                    b"pStyle" => {
                        // Heading style, robust to variants like "Heading1", "Heading 1", case-insensitive
                        if let Some(mut val) = attr_val(&e, b"val") {
                            let lower = val.to_ascii_lowercase();
                            if let Some(rest) = lower.strip_prefix("heading") {
                                let digits: String = rest.chars().filter(|ch| ch.is_ascii_digit()).collect();
                                if let Ok(n) = digits.parse::<u8>() { pending_heading_level = Some(n.max(1)); }
                            }
                            // Resolve via styles.xml when available (styleId mapping)
                            if pending_heading_level.is_none() {
                                if let Some(&lvl) = style_levels.get(&val) { pending_heading_level = Some(lvl); }
                            }
                            // Remember style id for style-based numbering fallback
                            current_style_id = Some(val);
                        }
                    }
                    b"outlineLvl" => {
                        // Paragraph outline level: 0 => Heading1, 1 => Heading2, ...
                        if let Some(vs) = attr_val(&e, b"val") {
                            if let Ok(n) = vs.parse::<u8>() { pending_heading_level = Some(n.saturating_add(1)); }
                        }
                    }
                    b"numId" => { if in_numpr { if let Some(v) = attr_val(&e, b"val") { pending_num_id = v.parse::<u32>().ok(); } } }
                    b"ilvl" => { if in_numpr { if let Some(v) = attr_val(&e, b"val") { pending_ilvl = v.parse::<u8>().ok(); } } }
                    b"t" => { in_t = true; }
                    b"br" => {
                        // page break or line break
                        if let Some(t) = attr_val(&e, b"type") {
                            if t.eq_ignore_ascii_case("page") { current_page = current_page.saturating_add(1); }
                        }
                        if in_tc { cell_text.push('\u{2028}'); } else { cur_text.push('\n'); }
                    }
                    b"tab" => { if in_tc { cell_text.push('\t'); } else { cur_text.push('\t'); } }
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => {
                match local_name(e.name().as_ref()) {
                    b"numPr" => { in_numpr = true; }
                    b"numId" => { if in_numpr { if let Some(v) = attr_val(&e, b"val") { pending_num_id = v.parse::<u32>().ok(); } } }
                    b"ilvl" => { if in_numpr { if let Some(v) = attr_val(&e, b"val") { pending_ilvl = v.parse::<u8>().ok(); } } }
                    b"pStyle" => {
                        if let Some(mut val) = attr_val(&e, b"val") {
                            let lower = val.to_ascii_lowercase();
                            if let Some(rest) = lower.strip_prefix("heading") {
                                let digits: String = rest.chars().filter(|ch| ch.is_ascii_digit()).collect();
                                if let Ok(n) = digits.parse::<u8>() { pending_heading_level = Some(n.max(1)); }
                            }
                            if pending_heading_level.is_none() {
                                if let Some(&lvl) = style_levels.get(&val) { pending_heading_level = Some(lvl); }
                            }
                            current_style_id = Some(val);
                        }
                    }
                    b"outlineLvl" => {
                        if let Some(vs) = attr_val(&e, b"val") {
                            if let Ok(n) = vs.parse::<u8>() { pending_heading_level = Some(n.saturating_add(1)); }
                        }
                    }
                    b"br" => {
                        if let Some(t) = attr_val(&e, b"type") {
                            if t.eq_ignore_ascii_case("page") { current_page = current_page.saturating_add(1); }
                        }
                        if in_tc { cell_text.push('\u{2028}'); } else { cur_text.push('\n'); }
                    }
                    b"tab" => { if in_tc { cell_text.push('\t'); } else { cur_text.push('\t'); } }
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                match local_name(e.name().as_ref()) {
                    b"t" => { in_t = false; }
                    b"p" => {
                        if in_tc {
                            // paragraph end inside table cell => treat as soft newline within the cell (U+2028)
                            if !cell_text.ends_with('\u{2028}') { cell_text.push('\u{2028}'); }
                        } else if in_p {
                            let mut text_owned = cur_text.trim().to_string();
                            // If paragraph has no explicit numPr, derive numbering from style
                            if pending_num_id.is_none() {
                                if let Some(sid) = &current_style_id {
                                    let style_numid = style_numpr.get(sid).and_then(|(n, _)| *n);
                                    let style_ilvl = style_numpr.get(sid).and_then(|(_, lv)| *lv);
                                    let pstyle_bind = numbering.style_default_num(sid); // (numId, ilvl) from numbering.xml pStyle binding

                                    match (style_numid, style_ilvl, pstyle_bind) {
                                        // Prefer style's numId with pStyle-derived ilvl when both available
                                        (Some(nid), _, Some((_pnid, pilvl))) => { pending_num_id = Some(nid); pending_ilvl = Some(pilvl); }
                                        // Fall back to style's explicit ilvl if present
                                        (Some(nid), Some(lv), None) => { pending_num_id = Some(nid); pending_ilvl = Some(lv); }
                                        // If style has no numId, use pStyle binding's numId+ilvl
                                        (None, _, Some((nid, lv))) => { pending_num_id = Some(nid); pending_ilvl = Some(lv); }
                                        _ => { /* leave unset; no further fallback to outline level here */ }
                                    }
                                }
                            }
                            // Apply numbering prefix when available
                            if let (Some(nid), Some(lv)) = (pending_num_id, pending_ilvl) {
                                let prefix = numbering.format_prefix(nid, lv);
                                if !prefix.is_empty() { text_owned = format!("{}{}", prefix, text_owned); }
                            }
                            if !text_owned.trim().is_empty() {
                                if let Some(level) = pending_heading_level {
                                    let mut b = UnifiedBlock::new(BlockKind::Heading, format!("{}\n", text_owned), order, path, "docx");
                                    b.heading_level = Some(level);
                                    b.page_start = Some(para_start_page);
                                    b.page_end = Some(current_page);
                                    blocks.push(b);
                                } else {
                                    let mut b = UnifiedBlock::new(BlockKind::Paragraph, text_owned, order, path, "docx");
                                    b.page_start = Some(para_start_page);
                                    b.page_end = Some(current_page);
                                    blocks.push(b);
                                }
                                order += 1;
                            }
                            in_p = false; cur_text.clear(); pending_heading_level = None; in_numpr=false; pending_num_id=None; pending_ilvl=None; current_style_id=None;
                        }
                    }
                    b"tc" => {
                        if in_tc {
                            in_tc = false;
                            // finalize a cell
                            let cell = cell_text.trim().to_string();
                            row_cells.push(cell);
                            cell_text.clear();
                        }
                    }
                    b"tr" => {
                        if in_tr {
                            in_tr = false;
                            // finalize a row => join cells with tabs, add newline
                            let line = row_cells.join("\t");
                            table_text.push_str(&line);
                            table_text.push('\n');
                            row_cells.clear();
                        }
                    }
                    b"tbl" => {
                        if in_tbl {
                            in_tbl = false;
                            // flush accumulated table text as a single block with HTML-like wrapper
                            if !table_text.is_empty() {
                                let mut wrapped = String::new();
                                // Add a leading newline only if previous block does not end with one
                                let need_leading_nl = blocks.last().map_or(false, |prev| !prev.text.ends_with('\n'));
                                if need_leading_nl { wrapped.push('\n'); }
                                let content = table_text.trim_end_matches('\n');
                                wrapped.push_str(&format!("<table delim=\"tsv\" cell-nl=\"U+2028\">\n{}\n</table>\n", content));
                                let mut b = UnifiedBlock::new(BlockKind::Paragraph, wrapped, order, path, "docx");
                                b.page_start = Some(table_start_page);
                                b.page_end = Some(current_page);
                                // optional hint that this is a table and the cell newline codepoint
                                b.attrs.insert("is_table".to_string(), "true".to_string());
                                b.attrs.insert("table_cell_nl".to_string(), "U+2028".to_string());
                                blocks.push(b);
                                order += 1;
                            }
                            table_text.clear();
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(t)) => {
                if in_t {
                    if let Ok(cow) = t.unescape() {
                        let s: String = cow.into_owned();
                        if in_tc { cell_text.push_str(&s); } else { cur_text.push_str(&s); }
                    }
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
