use crate::unified_blocks::{UnifiedBlock, BlockKind};
use calamine::Reader; // brings sheet_names/worksheet_range into scope

/// Minimal Excel reader using `calamine` to convert sheets and rows into `UnifiedBlock`s.
/// Formats: XLSX / XLS / ODS (auto-detected by calamine).
/// - Inserts a Heading block per sheet (page = sheet index starting at 1)
/// - Emits Paragraph blocks per row with tab-separated cell values
/// - Trims trailing empty cells per row and skips fully empty lines
pub fn read_excel_to_blocks(path: &str) -> Vec<UnifiedBlock> {
    let mut blocks: Vec<UnifiedBlock> = Vec::new();

    let mut workbook = match calamine::open_workbook_auto(path) {
        Ok(wb) => wb,
        Err(_) => return vec![UnifiedBlock::new(BlockKind::Paragraph, "(error) failed to open workbook", 0, path, "excel")],
    };

    let names: Vec<String> = workbook.sheet_names();
    if names.is_empty() {
        return vec![UnifiedBlock::new(BlockKind::Paragraph, "(empty workbook)", 0, path, "excel")];
    }

    let mut order: u32 = 0;
    for (sidx, name) in names.iter().enumerate() {
        let page = (sidx as u32) + 1;
        // Sheet heading
        {
            let mut h = UnifiedBlock::new(BlockKind::Heading, format!("Sheet: {}", name), order, path, "excel");
            h.heading_level = Some(1);
            h.page_start = Some(page);
            h.page_end = Some(page);
            blocks.push(h);
            order += 1;
        }

        let range = match workbook.worksheet_range(name) {
            Ok(r) => r,
            _ => {
                let mut b = UnifiedBlock::new(BlockKind::Paragraph, "(error) failed to read sheet".to_string(), order, path, "excel");
                b.page_start = Some(page);
                b.page_end = Some(page);
                blocks.push(b);
                order += 1;
                continue;
            }
        };

        for row in range.rows() {
            // Convert cells to strings and trim trailing empties
            let mut cells: Vec<String> = row.iter().map(cell_to_string).collect();
            while let Some(last) = cells.last() { if last.trim().is_empty() { cells.pop(); } else { break; } }
            if cells.is_empty() { continue; }
            let line = cells.join("\t");
            let text = line.trim();
            if text.is_empty() { continue; }
            let mut p = UnifiedBlock::new(BlockKind::Paragraph, text.to_string(), order, path, "excel");
            p.page_start = Some(page);
            p.page_end = Some(page);
            blocks.push(p);
            order += 1;
        }
    }

    if blocks.is_empty() {
        blocks.push(UnifiedBlock::new(BlockKind::Paragraph, String::new(), 0, path, "excel"));
    }
    blocks
}

fn cell_to_string(c: &calamine::DataType) -> String {
    use calamine::DataType as D;
    match c {
        D::Empty => String::new(),
        D::String(s) => s.trim().to_string(),
        D::Float(f) => {
            if f.fract() == 0.0 { format!("{}", *f as i64) } else { f.to_string() }
        }
        D::Int(i) => i.to_string(),
        D::Bool(b) => if *b { "TRUE".into() } else { "FALSE".into() },
        D::Error(e) => format!("#ERR:{:?}", e),
        // Best-effort for date/time/duration or any future variants
        other => format!("{}", other),
    }
}
