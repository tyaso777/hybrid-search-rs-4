use crate::unified_blocks::{UnifiedBlock, BlockKind, BBox, BBoxUnit};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfBackend {
    Stub,
    PureRust,
    Pdfium,
}

/// Select the default backend based on enabled cargo features.
pub fn default_backend() -> PdfBackend {
    #[cfg(feature = "pdfium")] { return PdfBackend::Pdfium; }
    #[cfg(all(not(feature = "pdfium"), feature = "pure-pdf"))] { return PdfBackend::PureRust; }
    #[cfg(all(not(feature = "pdfium"), not(feature = "pure-pdf")))] { return PdfBackend::Stub; }
}

pub fn read_pdf_to_blocks_with(path: &str, backend: PdfBackend) -> Vec<UnifiedBlock> {
    match backend {
        PdfBackend::Stub => read_pdf_stub(path),
        PdfBackend::PureRust => {
            #[cfg(feature = "pure-pdf")]
            {
                return crate::reader_pdf_pure::read_pdf_to_blocks_pure(path);
            }
            #[allow(unreachable_code)]
            {
                eprintln!("pure-pdf backend not enabled; falling back to stub");
                read_pdf_stub(path)
            }
        }
        PdfBackend::Pdfium => {
            #[cfg(feature = "pdfium")]
            {
                return crate::reader_pdf_pdfium::read_pdf_to_blocks_pdfium(path);
            }
            #[allow(unreachable_code)]
            {
                eprintln!("pdfium backend not enabled; falling back to stub");
                read_pdf_stub(path)
            }
        }
    }
}

/// Stub PDF reader that returns a single block placeholder.
pub fn read_pdf_to_blocks(path: &str) -> Vec<UnifiedBlock> {
    read_pdf_to_blocks_with(path, default_backend())
}

fn read_pdf_stub(path: &str) -> Vec<UnifiedBlock> {
    // For now, create a few sample blocks to visualize the contract.
    let reader = "stub.pdf";
    let mut blocks = Vec::new();

    let mut h1 = UnifiedBlock::new(BlockKind::Heading, "第1章 概要", 0, path, reader);
    h1.page_start = Some(1);
    h1.page_end = Some(1);
    h1.heading_level = Some(1);
    h1.section_hint = Some(crate::unified_blocks::SectionHint { level: 1, title: "概要".into(), numbering: Some("第1章".into()) });
    h1.bbox = Some(BBox { x: 72.0, y: 72.0, w: 468.0, h: 24.0, unit: BBoxUnit::Pt });
    blocks.push(h1);

    // Paragraph spanning two pages
    let mut p1 = UnifiedBlock::new(
        BlockKind::Paragraph,
        "これはPDFから抽出されたテキストのサンプルです。複数ページにまたがる場合があります。",
        1,
        path,
        reader,
    );
    p1.page_start = Some(1);
    p1.page_end = Some(2);
    p1.lang = Some("ja".into());
    p1.bbox = Some(BBox { x: 72.0, y: 120.0, w: 468.0, h: 64.0, unit: BBoxUnit::Pt });
    blocks.push(p1);

    let mut li = UnifiedBlock::new(BlockKind::ListItem, "・箇条書き項目の例", 2, path, reader);
    li.page_start = Some(2);
    li.page_end = Some(2);
    li.list = Some(crate::unified_blocks::ListInfo { ordered: false, level: 1, marker: Some("・".into()) });
    blocks.push(li);

    let mut h2 = UnifiedBlock::new(BlockKind::Heading, "1.1 詳細", 3, path, reader);
    h2.page_start = Some(2);
    h2.page_end = Some(2);
    h2.heading_level = Some(2);
    h2.section_hint = Some(crate::unified_blocks::SectionHint { level: 2, title: "詳細".into(), numbering: Some("1.1".into()) });
    blocks.push(h2);

    let mut code = UnifiedBlock::new(BlockKind::Code, "fn main() { println!(\"hello\"); }", 4, path, reader);
    code.page_start = Some(2);
    code.page_end = Some(2);
    code.code_lang = Some("rust".into());
    blocks.push(code);

    blocks
}
