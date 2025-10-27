use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

/// Block structure kind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockKind {
    Paragraph,
    Heading,
    ListItem,
    Code,
    TableCell,
    FigureCaption,
    Header,
    Footer,
    PageBreak,
}

/// List item info when kind == ListItem.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListInfo {
    pub ordered: bool,
    pub level: u8,
    pub marker: Option<String>,
}

/// Table cell info when kind == TableCell.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TableInfo {
    pub row: u32,
    pub col: u32,
    pub span: Option<(u32, u32)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BBoxUnit { Pt, Norm }

/// Bounding box of the block when available.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub unit: BBoxUnit,
}

/// Reader/source info.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceRef {
    pub reader: String,
    pub origin: String,
    pub local_id: String,
}

/// Section hint for later section_path generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionHint {
    pub level: u8,
    pub title: String,
    pub numbering: Option<String>,
}

/// Link range within text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkRef {
    pub start: u32,
    pub end: u32,
    pub uri: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeaderFooterHint { Header, Footer }

/// A unified representation of sequential text blocks from various readers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UnifiedBlock {
    /// Normalized text (preserve newlines; remove control chars lightly).
    pub text: String,
    /// Structural kind of the block.
    pub kind: BlockKind,
    /// Page span (1-based) if available. Blocks may cross pages.
    pub page_start: Option<u32>,
    pub page_end: Option<u32>,
    /// Global reading order (monotonic across document).
    pub order: u32,
    /// Heading level when kind == Heading.
    pub heading_level: Option<u8>,
    /// List item info when kind == ListItem.
    pub list: Option<ListInfo>,
    /// Table cell info when kind == TableCell.
    pub table: Option<TableInfo>,
    /// Bounding box when available.
    pub bbox: Option<BBox>,
    /// Language hint (e.g., "ja").
    pub lang: Option<String>,
    /// Source/reader info.
    pub source: SourceRef,
    /// Forward-compatible attributes (style/layout/etc.).
    pub attrs: BTreeMap<String, String>,

    /// Optional: section heading hint for later path creation.
    pub section_hint: Option<SectionHint>,
    /// Optional: code language when kind == Code.
    pub code_lang: Option<String>,
    /// Optional: embedded links in text.
    pub links: Vec<LinkRef>,
    /// Optional: extraction confidence (e.g., OCR).
    pub confidence: Option<f32>,
    /// Optional: header/footer hint for repeated patterns.
    pub header_footer_hint: Option<HeaderFooterHint>,
}

impl UnifiedBlock {
    /// Convenience constructor.
    pub fn new(kind: BlockKind, text: impl Into<String>, order: u32, origin: impl Into<String>, reader: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            kind,
            page_start: None,
            page_end: None,
            order,
            heading_level: None,
            list: None,
            table: None,
            bbox: None,
            lang: None,
            source: SourceRef { reader: reader.into(), origin: origin.into(), local_id: String::new() },
            attrs: BTreeMap::new(),
            section_hint: None,
            code_lang: None,
            links: Vec::new(),
            confidence: None,
            header_footer_hint: None,
        }
    }
}
