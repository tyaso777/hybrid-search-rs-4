use eframe::egui;
use std::env;
use std::fs;
use std::path::PathBuf;
use file_chunker::reader_pdf;
use file_chunker::unified_blocks::UnifiedBlock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UiBackend {
    Auto,
    Stub,
    PureRust,
    Pdfium,
}

struct AppState {
    pdf_path: String,
    blocks: Vec<UnifiedBlock>,
    error: Option<String>,
    backend: UiBackend,
    show_tab_escape: bool,
}

impl Default for AppState {
    fn default() -> Self {
        // Default to PDFium backend for better extraction (DLL required).
        Self {
            pdf_path: String::new(),
            blocks: Vec::new(),
            error: None,
            backend: UiBackend::Pdfium,
            show_tab_escape: true,
        }
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "PDF UnifiedBlocks Viewer",
        options,
        Box::new(|cc| {
            install_japanese_fallback_fonts(&cc.egui_ctx);
            Box::new(AppState::default())
        }),
    )
}

impl eframe::App for AppState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("PDF path:");
                ui.text_edit_singleline(&mut self.pdf_path);
                if ui.button("Pick...").clicked() {
                    if let Some(path) = rfd::FileDialog::new().add_filter("PDF", &["pdf"]).pick_file() {
                        self.pdf_path = path.display().to_string();
                    }
                }
                ui.separator();
                ui.label("Backend:");
                egui::ComboBox::from_label("")
                    .selected_text(match self.backend { UiBackend::Auto => "Auto", UiBackend::Stub => "Stub", UiBackend::PureRust => "Pure (lopdf)", UiBackend::Pdfium => "PDFium" })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.backend, UiBackend::Auto, "Auto");
                        ui.selectable_value(&mut self.backend, UiBackend::Stub, "Stub");
                        ui.selectable_value(&mut self.backend, UiBackend::PureRust, "Pure (lopdf)");
                        ui.selectable_value(&mut self.backend, UiBackend::Pdfium, "PDFium");
                    });
                if ui.button("Parse").clicked() {
                    let path = self.pdf_path.trim();
                    if path.is_empty() {
                        self.error = Some("Please select a PDF path".into());
                        self.blocks.clear();
                    } else {
                        self.error = None;
                        // Dispatch to selected backend.
                        self.blocks = match self.backend {
                            UiBackend::Auto => reader_pdf::read_pdf_to_blocks(path),
                            UiBackend::Stub => reader_pdf::read_pdf_to_blocks_with(path, reader_pdf::PdfBackend::Stub),
                            UiBackend::PureRust => reader_pdf::read_pdf_to_blocks_with(path, reader_pdf::PdfBackend::PureRust),
                            UiBackend::Pdfium => reader_pdf::read_pdf_to_blocks_with(path, reader_pdf::PdfBackend::Pdfium),
                        };
                    }
                }
                ui.separator();
                ui.checkbox(&mut self.show_tab_escape, "Show \t for tabs");
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(err) = &self.error { ui.colored_label(egui::Color32::RED, err); }

            ui.label(format!("Blocks: {}", self.blocks.len()));
            ui.separator();

            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Features:");
                    ui.monospace(format!("pdfium={} pure-pdf={}", cfg!(feature = "pdfium"), cfg!(feature = "pure-pdf")));
                    ui.separator();
                    ui.label("Default backend:");
                    let default = if cfg!(feature = "pdfium") { "PDFium" } else if cfg!(feature = "pure-pdf") { "Pure (lopdf)" } else { "Stub" };
                    ui.monospace(default);
                });
                ui.separator();
                for (i, b) in self.blocks.iter().enumerate() {
                    ui.collapsing(format!("Block {i} — {:?}", b.kind), |ui| {
                        // Escaped text preview (tabs as \t when enabled)
                        let preview = escape_text_for_view(&b.text, self.show_tab_escape);
                        ui.label("Text (preview):");
                        ui.monospace(preview);
                        ui.separator();
                        let json = serde_json::to_string_pretty(b).unwrap_or_else(|_| "<serde error>".into());
                        ui.monospace(json);
                    });
                }
            });
        });
    }
}

fn escape_text_for_view(s: &str, show_tab_escape: bool) -> String {
    if !show_tab_escape { return s.to_string(); }
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\t' => out.push_str("\\t"),
            '\r' => { /* drop CR for stable display */ }
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

    // 1) Explicit override via environment variable
    if let Ok(custom) = env::var("PDF_BLOCK_VIEWER_FONT") {
        paths.push(PathBuf::from(custom));
    }
    // 1b) Fallback to workspace-wide demo env
    if let Ok(custom) = env::var("EMBEDDER_DEMO_FONT") {
        paths.push(PathBuf::from(custom));
    }

    // 2) Windows common fonts
    if let Ok(windir) = env::var("WINDIR") {
        let fonts_dir = PathBuf::from(windir).join("Fonts");
        for candidate in [
            "YuGothM.ttc",
            "YuGothB.ttc",
            "meiryo.ttc",
            "msgothic.ttc",
        ] {
            paths.push(fonts_dir.join(candidate));
        }
    }

    // 3) macOS common fonts
    for candidate in [
        "/System/Library/Fonts/Hiragino Sans W3.ttc",
        "/System/Library/Fonts/Hiragino Sans W6.ttc",
        "/Library/Fonts/Osaka.ttf",
    ] {
        paths.push(PathBuf::from(candidate));
    }

    // 4) Linux common Noto CJK locations
    for candidate in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ] {
        paths.push(PathBuf::from(candidate));
    }

    // 5) Project-local fallback (if available)
    paths.push(PathBuf::from("fonts/NotoSansJP-Regular.otf"));

    paths
}
