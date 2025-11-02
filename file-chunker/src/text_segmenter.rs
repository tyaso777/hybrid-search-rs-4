use crate::unified_blocks::UnifiedBlock;

#[derive(Debug, Clone, Copy)]
pub struct TextChunkParams {
    pub min_chars: usize,
    pub max_chars: usize,
    pub cap_chars: usize,
    /// Penalize cutting immediately after a very short line.
    pub penalize_short_line: bool,
    /// Penalize cutting at a page boundary when there is no newline before.
    pub penalize_page_boundary_no_newline: bool,
}

impl Default for TextChunkParams {
    fn default() -> Self {
        Self { min_chars: 400, max_chars: 600, cap_chars: 800, penalize_short_line: true, penalize_page_boundary_no_newline: true }
    }
}

#[derive(Debug, Clone, Copy)]
struct Boundary { idx: usize, base_score: f32 }

#[derive(Debug, Clone, Copy)]
struct BlockSpan { start: usize, end: usize, page_start: Option<u32>, page_end: Option<u32> }

fn collect_text_and_boundaries(blocks: &[UnifiedBlock]) -> (String, Vec<Boundary>, Vec<BlockSpan>) {
    let mut text = String::new();
    let mut boundaries: Vec<Boundary> = Vec::new();
    let mut spans: Vec<BlockSpan> = Vec::new();

    let mut cursor = 0usize;
    for (i, b) in blocks.iter().enumerate() {
        let t = b.text.replace('\r', "");
        let start_idx = cursor;
        text.push_str(&t);
        cursor += t.len();
        spans.push(BlockSpan { start: start_idx, end: cursor, page_start: b.page_start, page_end: b.page_end });
        if i + 1 < blocks.len() {
            // Prefer block boundaries strongly
            boundaries.push(Boundary { idx: cursor, base_score: 1.0 });
        }
    }

    // Newline boundaries
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'\n' {
            // Double newline?
            if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                boundaries.push(Boundary { idx: i + 2, base_score: 0.95 });
                i += 2;
                continue;
            } else {
                boundaries.push(Boundary { idx: i + 1, base_score: 0.8 });
            }
        }
        i += 1;
    }

    // Sentence ends for ASCII and Japanese punctuation (exclude dot leaders as hard boundaries)
    for (idx, ch) in text.char_indices() {
        if matches!(ch, '.' | '!' | '?' | '。' | '！' | '？') {
            boundaries.push(Boundary { idx: idx + ch.len_utf8(), base_score: 0.6 });
        }
    }

    // (moved is_leader_char helper into chunking function)

    // Sort and dedup by idx (keep highest base_score)
    boundaries.sort_by_key(|b| b.idx);
    boundaries.dedup_by(|a, b| {
        if a.idx == b.idx {
            if a.base_score < b.base_score { a.base_score = b.base_score; }
            true
        } else { false }
    });

    (text, boundaries, spans)
}

fn penalize_after_short_line(text: &str, idx: usize) -> f32 {
    // Penalize if immediately after a very short line (encourage merging)
    let mut j = if idx > 0 { idx - 1 } else { 0 };
    while j > 0 && text.as_bytes()[j] != b'\n' { j -= 1; }
    let line_start = if text.as_bytes()[j] == b'\n' { j + 1 } else { j };
    let line_len = idx.saturating_sub(line_start);
    if line_len < 10 { 0.35 } else { 0.0 }
}

fn extra_penalty_page_boundary_no_newline(idx: usize, text: &str, spans: &[BlockSpan]) -> f32 {
    for w in spans.windows(2) {
        let a = &w[0];
        let b = &w[1];
        if a.end == idx {
            let page_transition = match (a.page_end, b.page_start) { (Some(pe), Some(ps)) => pe != ps, _ => false };
            let has_newline_before = idx > 0 && text.as_bytes()[idx.saturating_sub(1)] == b'\n';
            if page_transition && !has_newline_before { return 0.4; }
        }
    }
    0.0
}

fn pick_boundary_in_range(scored: &[(usize, f32)], lo: usize, hi: usize, prefer: usize) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (idx, score) in scored {
        if *idx < lo || *idx > hi { continue; }
        let dist = if *idx > prefer { *idx - prefer } else { prefer - *idx } as f32;
        let span = (hi.saturating_sub(lo)) as f32 + 1.0;
        let eff = *score - dist / span;
        if let Some((_, b)) = best { if eff > b { best = Some((*idx, eff)); } } else { best = Some((*idx, eff)); }
    }
    best.map(|(i, _)| i)
}

fn page_range_for_segment(start: usize, end: usize, spans: &[BlockSpan]) -> (Option<u32>, Option<u32>) {
    let mut min_p: Option<u32> = None;
    let mut max_p: Option<u32> = None;
    for s in spans {
        if s.end <= start || s.start >= end { continue; }
        if let Some(ps) = s.page_start { min_p = Some(match min_p { Some(v) => v.min(ps), None => ps }); }
        if let Some(pe) = s.page_end { max_p = Some(match max_p { Some(v) => v.max(pe), None => pe }); }
    }
    (min_p, max_p)
}

/// Generic block-to-segments chunker shared by PDF/TXT/etc.
pub fn chunk_blocks_to_segments(blocks: &[UnifiedBlock], params: &TextChunkParams) -> Vec<(String, Option<u32>, Option<u32>)> {
    let (text, boundaries, spans) = collect_text_and_boundaries(blocks);
    if text.trim().is_empty() { return vec![(String::new(), None, None)]; }

    // Score boundaries with optional penalties
    // Helper: leader characters used for TOC dot leaders
    let is_leader_char = |c: char| matches!(c, '.' | '…' | '・');

    // Detect table block spans in concatenated text (prefer attrs flag; fallback to text prefix)
    let mut table_ranges: Vec<(usize, usize)> = Vec::new();
    for (i, b) in blocks.iter().enumerate() {
        let is_table = b.attrs.get("is_table").map(|v| v == "true").unwrap_or_else(|| b.text.trim_start().starts_with("<table "));
        if is_table {
            if let Some(span) = spans.get(i) { table_ranges.push((span.start, span.end)); }
        }
    }
    table_ranges.sort_by_key(|r| r.0);

    let mut scored: Vec<(usize, f32)> = boundaries.iter().map(|b| {
        let mut s = b.base_score;
        if params.penalize_short_line { s -= penalize_after_short_line(&text, b.idx); }
        if params.penalize_page_boundary_no_newline { s -= extra_penalty_page_boundary_no_newline(b.idx, &text, &spans); }
        // Penalize boundaries that fall inside a dot-leader run (e.g., "……")
        // and avoid cutting immediately after a leader char.
        // Compute leader run lengths around the boundary: consecutive leaders to the left/right.
        let mut left_len = 0usize;
        {
            let mut pos = b.idx;
            while pos > 0 {
                // find prev char start
                let mut p = pos - 1; while p > 0 && !text.is_char_boundary(p) { p -= 1; }
                if !text.is_char_boundary(p) { break; }
                if let Some(ch) = text[p..pos].chars().next() { if is_leader_char(ch) { left_len += 1; pos = p; continue; } }
                break;
            }
        }
        let mut right_len = 0usize;
        {
            let mut pos = b.idx;
            while pos < text.len() {
                // find next char end
                if !text.is_char_boundary(pos) { break; }
                if let Some(ch) = text[pos..].chars().next() {
                    let next = pos + ch.len_utf8();
                    if is_leader_char(ch) { right_len += 1; pos = next; continue; }
                }
                break;
            }
        }
        let run_len = left_len + right_len;
        if run_len >= 3 {
            if left_len > 0 {
                // inside or just after leader run -> strong penalty
                s -= 0.6;
            } else {
                // boundary is just before a leader run; keep neutral (no bonus) to avoid preferring it too much
            }
        }
        (b.idx, s)
    }).collect();
    scored.sort_by_key(|p| p.0);

    // Remove boundaries that fall inside table blocks (keep edges)
    scored.retain(|(idx, _)| {
        !table_ranges.iter().any(|(s, e)| *idx > *s && *idx < *e)
    });

    // Table-aware score tweaks:
    // - discourage cutting right before a table (prefer to keep the previous sentence with the table)
    // - slightly encourage cutting right after a table
    if !table_ranges.is_empty() {
        let idxs: Vec<usize> = scored.iter().map(|p| p.0).collect();
        for (ts, te) in &table_ranges {
            // reward boundary at table end (block boundary)
            if let Ok(pos) = idxs.binary_search(te) {
                if let Some((_i, score)) = scored.get_mut(pos) { *score += 0.25; }
            }
            // find last two boundaries before table start
            match idxs.binary_search(ts) {
                Ok(pos) | Err(pos) => {
                    if pos > 0 {
                        let prev = pos - 1; // last boundary before table
                        if let Some((_i, score)) = scored.get_mut(prev) { *score -= 0.3; }
                        if prev > 0 {
                            let prev2 = prev - 1; // one more boundary earlier
                            if let Some((_i, score)) = scored.get_mut(prev2) { *score += 0.15; }
                        }
                    }
                }
            }
        }
    }

    let total = text.len();
    let mut start = 0usize;
    let mut out: Vec<(String, Option<u32>, Option<u32>)> = Vec::new();
    while start < total {
        let min = start.saturating_add(params.min_chars.min(total - start));
        let max = start.saturating_add(params.max_chars.min(total - start));
        let cap = start.saturating_add(params.cap_chars.min(total - start));
        // Ensure we have a valid char boundary for a hard cap fallback
        let mut hard_cap = cap;
        while hard_cap > start && !text.is_char_boundary(hard_cap) { hard_cap -= 1; }
        if hard_cap <= start { hard_cap = (cap + 1).min(total); while hard_cap < total && !text.is_char_boundary(hard_cap) { hard_cap += 1; } }
        // Avoid placing hard cap inside a leader run: move left to the start of the run when detected.
        {
            let mut pos = hard_cap;
            let mut moved = false;
            // step back while previous char is a leader
            loop {
                if pos == start { break; }
                let mut p = pos - 1; while p > 0 && !text.is_char_boundary(p) { p -= 1; }
                if !text.is_char_boundary(p) { break; }
                if let Some(ch) = text[p..pos].chars().next() {
                    if is_leader_char(ch) { pos = p; moved = true; continue; }
                }
                break;
            }
            if moved { hard_cap = pos.max(start + 1); }
        }

        if start + params.min_chars >= total {
            let seg = text[start..total].trim_end();
            if !seg.is_empty() {
                let (ps, pe) = page_range_for_segment(start, total, &spans);
                out.push((seg.to_string(), ps, pe));
            }
            break;
        }

        // Table-aware adjustment before finalizing cut selection
        // Identify the next table that starts at/after current start
        let next_table = table_ranges.iter().find(|(s, _e)| *s >= start);

        if let Some(mut cut) = pick_boundary_in_range(&scored, min, cap, max) {
            if cut > start {
                if let Some((ts, te)) = next_table.copied() {
                    // If we are about to cut before the table while the table (and text until its end)
                    // still fits under cap, try to cut earlier so that the next segment will include
                    // the sentence before the table together with the table.
                    if cut <= ts && te <= cap {
                        // find last two boundaries before table start
                        let idxs: Vec<usize> = scored.iter().map(|p| p.0).collect();
                        let mut prev2: Option<usize> = None;
                        match idxs.binary_search(&ts) {
                            Ok(pos) | Err(pos) => {
                                if pos > 1 { prev2 = Some(idxs[pos - 2]); }
                                else if pos > 0 { prev2 = Some(idxs[pos - 1]); }
                            }
                        }
                        if let Some(b2) = prev2 { if b2 >= min { cut = b2; } }
                    }
                }
                let seg = text[start..cut].trim_end();
                if !seg.is_empty() {
                    let (ps, pe) = page_range_for_segment(start, cut, &spans);
                    out.push((seg.to_string(), ps, pe));
                }
                start = cut;
                continue;
            }
        }

        // Fallback: boundary after cap, else last boundary
        let mut fallback_cut: Option<usize> = None;
        for (idx, _) in &scored { if *idx > cap { fallback_cut = Some(*idx); break; } }
        if fallback_cut.is_none() { if let Some((idx, _)) = scored.last() { fallback_cut = Some(*idx); } }
        // Enforce hard cap if no reasonable boundary is found
        let mut cut = fallback_cut.unwrap_or(hard_cap);
        if cut > hard_cap { cut = hard_cap; }
        if cut <= start { cut = hard_cap; }
        if cut <= start { cut = total; } // safety to avoid infinite loop
        let seg = text[start..cut].trim_end();
        if !seg.is_empty() {
            let (ps, pe) = page_range_for_segment(start, cut, &spans);
            out.push((seg.to_string(), ps, pe));
        }
        start = cut;
    }

    if out.is_empty() { out.push((String::new(), None, None)); }
    // Post-process: merge too-short trailing segments into the previous one when it doesn't exceed cap.
    // This helps avoid very small chunks caused by hard boundaries.
    let mut merged: Vec<(String, Option<u32>, Option<u32>)> = Vec::with_capacity(out.len());
    let short_min: usize = 100; // characters
    for (text, ps, pe) in out.into_iter() {
        if let Some((prev_text, prev_ps, prev_pe)) = merged.last_mut() {
            let curr_len = text.chars().count();
            let prev_len = prev_text.chars().count();
            if curr_len <= short_min && prev_len + curr_len <= params.cap_chars {
                // Join with a newline separator if previous doesn't already end with one (it shouldn't due to trim_end)
                if !prev_text.ends_with('\n') { prev_text.push('\n'); }
                prev_text.push_str(&text);
                // Merge page ranges
                let start_page = match (*prev_ps, ps) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    _ => None,
                };
                let end_page = match (*prev_pe, pe) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    _ => None,
                };
                *prev_ps = start_page;
                *prev_pe = end_page;
                continue;
            }
        }
        merged.push((text, ps, pe));
    }
    if merged.is_empty() { merged.push((String::new(), None, None)); }
    merged
}
