use eframe::egui;
use egui_extras::{StripBuilder, Size};
use std::env;
use std::fs;
use std::path::PathBuf;

use chunk_model::{ChunkRecord, FileRecord, DocumentId, ChunkId, SCHEMA_MAJOR};
use file_chunker::pdf_chunker::PdfChunkParams;

#[derive(Debug, Default)]
struct AppState {
    path: String,
    params: PdfChunkParams,
    gap_chars: usize,
    file_json: String,
    chunks: Vec<ChunkRecord>,
    error: Option<String>,
    selected: Option<usize>,
    show_tab_escape: bool,
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "Chunker Viewer",
        options,
        Box::new(|cc| {
            install_japanese_fallback_fonts(&cc.egui_ctx);
            Box::new(AppState { show_tab_escape: true, gap_chars: 3, ..Default::default() })
        }),
    )
}

impl eframe::App for AppState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.vertical(|ui| {
                // Row 1: file input + actions
                ui.horizontal(|ui| {
                    ui.label("File:");
                    ui.text_edit_singleline(&mut self.path);
                    if ui.button("Pick...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Documents", &["pdf", "docx", "txt", "xlsx", "pptx"]).pick_file()
                        {
                            self.path = path.display().to_string();
                        }
                    }
                    ui.separator();
                    if ui.button("Chunk").clicked() {
                        self.error = None;
                        self.file_json.clear();
                        self.chunks.clear();
                        self.selected = None;
                        let path = self.path.trim();
                        if path.is_empty() {
                            self.error = Some("Please pick a file".into());
                        } else {
                            #[cfg(feature = "pdfium")]
                            {
                                file_chunker::reader_pdf_pdfium::set_line_gap_chars(self.gap_chars);
                                // geometry params call removed                                // geometry params call removed
                            }
                            match chunk_file_auto(path, self.params) {
                                Ok((f, chunks)) => {
                                    self.file_json = serde_json::to_string_pretty(&f)
                                        .unwrap_or_else(|_| "<serde error>".into());
                                    self.chunks = chunks;
                                }
                                Err(e) => {
                                    self.error = Some(e);
                                }
                            }
                        }
                    }
                    ui.separator();
                    ui.checkbox(&mut self.show_tab_escape, "Show \\t for tabs");
                });

                // Row 2: params (on a separate line)
                ui.horizontal_wrapped(|ui| {
                    ui.label("Params:");
                    ui.add(egui::DragValue::new(&mut self.params.min_chars).clamp_range(1..=20000).prefix("min(chars)=").speed(10));
                    ui.add(egui::DragValue::new(&mut self.params.max_chars).clamp_range(1..=20000).prefix("max(chars)=").speed(10));
                    ui.add(egui::DragValue::new(&mut self.params.cap_chars).clamp_range(1..=20000).prefix("cap(chars)=").speed(10));
                    ui.separator();
                    ui.label("break-gap(c):");
                    ui.add(egui::DragValue::new(&mut self.gap_chars).clamp_range(1..=100).speed(1));
                    
                });
            });
        });

        egui::SidePanel::left("left").resizable(true).show(ctx, |ui| {
            ui.heading("FileRecord");
            if let Some(err) = &self.error {
                ui.colored_label(egui::Color32::RED, err);
                ui.separator();
            }
            if self.file_json.is_empty() {
                ui.label("No file loaded.");
            } else {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.monospace(&self.file_json);
                });
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            StripBuilder::new(ui)
                .size(Size::relative(0.55)) // top list initial share
                .size(Size::remainder())    // bottom selected chunk
                .clip(true)
                .vertical(|mut strip| {
                    strip.cell(|ui| {
                        ui.heading(format!("Chunks ({}):", self.chunks.len()));
                        ui.separator();
                        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                            for (i, c) in self.chunks.iter().enumerate() {
                                let preview = truncate_chars(&c.text, 80);
                                let page_label = pages_label(c);
                                let title = if page_label.is_empty() {
                                    format!("{}  {}", c.chunk_id.0, preview)
                                } else {
                                    format!("{}  {}  {}", c.chunk_id.0, page_label, preview)
                                };
                                let click = ui.selectable_label(self.selected == Some(i), title);
                                if click.clicked() { self.selected = Some(i); }
                            }
                        });
                    });
                    strip.cell(|ui| {
                        egui::Frame::default().show(ui, |ui| {
                            ui.heading("Selected Chunk");
                            ui.separator();
                            if let Some(i) = self.selected { if let Some(c) = self.chunks.get(i) {
                                let text = if self.show_tab_escape { escape_text_for_view(&c.text) } else { c.text.clone() };
                                let page_label = pages_label(c);
                                if page_label.is_empty() {
                                    ui.monospace(format!("len={} bytes", c.text.len()));
                                } else {
                                    ui.monospace(format!("len={} bytes  |  {}", c.text.len(), page_label));
                                }
                                ui.separator();
                                egui::ScrollArea::vertical().show(ui, |ui| {
                                    ui.monospace(text);
                                });
                            }}
                        });
                    });
                });
        });
    }
}

fn chunk_file_auto(path: &str, params: PdfChunkParams) -> Result<(FileRecord, Vec<ChunkRecord>), String> {
    let lower = path.to_lowercase();
    if lower.ends_with(".pdf") {
        Ok(file_chunker::pdf_chunker::chunk_pdf_file_with_file_record(path, &params))
    } else if lower.ends_with(".txt") {
        // Use the same segmentation parameters for TXT via text_segmenter
        let blocks = file_chunker::reader_txt::read_txt_to_blocks(path);
        let tparams = file_chunker::text_segmenter::TextChunkParams {
            min_chars: params.min_chars,
            max_chars: params.max_chars,
            cap_chars: params.cap_chars,
            penalize_short_line: true,
            penalize_page_boundary_no_newline: true,
        };
        let segs = file_chunker::text_segmenter::chunk_blocks_to_segments(&blocks, &tparams);

        let file = FileRecord {
            schema_version: SCHEMA_MAJOR,
            doc_id: DocumentId(path.to_string()),
            doc_revision: Some(1),
            source_uri: path.to_string(),
            source_mime: "text/plain".into(),
            file_size_bytes: None,
            content_sha256: None,
            page_count: None,
            extracted_at: String::new(),
            created_at_meta: None,
            updated_at_meta: None,
            title_guess: None,
            author_guess: None,
            dominant_lang: None,
            tags: Vec::new(),
            ingest_tool: Some("txt-chunker".into()),
            ingest_tool_version: Some(env!("CARGO_PKG_VERSION").into()),
            reader_backend: Some("txt".into()),
            ocr_used: None,
            ocr_langs: Vec::new(),
            chunk_count: Some(segs.len() as u32),
            total_tokens: None,
            meta: std::collections::BTreeMap::new(),
            extra: std::collections::BTreeMap::new(),
        };

        let chunks: Vec<ChunkRecord> = segs
            .into_iter()
            .enumerate()
            .map(|(i, (text, _ps, _pe))| ChunkRecord {
                schema_version: SCHEMA_MAJOR,
                doc_id: DocumentId(path.to_string()),
                chunk_id: ChunkId(format!("{}#{}", path, i)),
                source_uri: path.to_string(),
                source_mime: "text/plain".into(),
                extracted_at: String::new(),
                page_start: None,
                page_end: None,
                text,
                section_path: None,
                meta: std::collections::BTreeMap::new(),
                extra: std::collections::BTreeMap::new(),
            })
            .collect();

        Ok((file, chunks))
    } else {
        // Use generic path for other types; future: extend for docx/txt/xlsx/pptx
        let out = file_chunker::chunk_file_with_file_record(path);
        Ok((out.file, out.chunks))
    }
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if max_chars == 0 { return String::new(); }
    let mut it = s.chars();
    let truncated: String = it.by_ref().take(max_chars).collect();
    if it.next().is_some() { format!("{}…", truncated) } else { truncated }
}

fn pages_label(c: &ChunkRecord) -> String {
    let ps = c.page_start;
    let pe = c.page_end;
    match (ps, pe) {
        (Some(s), Some(e)) if s == e => format!("{}", s),
        (Some(s), Some(e)) => format!("{}-{}", s, e),
        (Some(s), None) => format!("{}", s),
        _ => String::new(),
    }
}

fn escape_text_for_view(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\t' => out.push_str("\\t"),
            '\r' => { /* skip */ }
            _ => out.push(ch),
        }
    }
    out
}

// Install Japanese font fallback so CJK renders correctly.
fn install_japanese_fallback_fonts(ctx: &egui::Context) {
    if let Some(data) = load_cjk_font_data() {
        let mut fonts = eframe::egui::FontDefinitions::default();
        fonts
            .font_data
            .insert("jp_fallback".into(), eframe::egui::FontData::from_owned(data));

        for family in [eframe::egui::FontFamily::Proportional, eframe::egui::FontFamily::Monospace]
        {
            fonts
                .families
                .entry(family)
                .or_default()
                .insert(0, "jp_fallback".into());
        }

        ctx.set_fonts(fonts);
    }
}

fn load_cjk_font_data() -> Option<Vec<u8>> {
    for path in candidate_font_paths() {
        if let Ok(data) = fs::read(&path) {
            return Some(data);
        }
    }
    None
}

fn candidate_font_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(custom) = env::var("CHUNKER_VIEWER_FONT") { paths.push(PathBuf::from(custom)); }
    if let Ok(custom) = env::var("EMBEDDER_DEMO_FONT") { paths.push(PathBuf::from(custom)); }

    if let Ok(windir) = env::var("WINDIR") {
        let fonts_dir = PathBuf::from(windir).join("Fonts");
        for candidate in ["YuGothM.ttc", "YuGothB.ttc", "meiryo.ttc", "msgothic.ttc"] {
            paths.push(fonts_dir.join(candidate));
        }
    }
    for candidate in [
        "/System/Library/Fonts/Hiragino Sans W3.ttc",
        "/System/Library/Fonts/Hiragino Sans W6.ttc",
        "/Library/Fonts/Osaka.ttf",
    ] { paths.push(PathBuf::from(candidate)); }
    for candidate in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ] { paths.push(PathBuf::from(candidate)); }
    paths.push(PathBuf::from("fonts/NotoSansJP-Regular.otf"));
    paths
}
