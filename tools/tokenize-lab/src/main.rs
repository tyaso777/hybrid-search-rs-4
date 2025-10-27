use eframe::egui::{self, Button, CentralPanel, ScrollArea, TextEdit};
use eframe::{App, CreationContext, Frame, NativeOptions};
use egui_extras::{Column, TableBuilder};
use std::env;
use std::fs;
use std::path::PathBuf;
use lindera::dictionary::load_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use lindera_tantivy::tokenizer::LinderaTokenizer;
use tantivy::schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions};
use tantivy::tokenizer::{NgramTokenizer, Tokenizer, TokenStream};
use tantivy::{Index, query::QueryParser};

fn main() -> eframe::Result<()> {
    let options = NativeOptions::default();
    eframe::run_native(
        "Tokenizer Lab",
        options,
        Box::new(|cc| Box::new(LabApp::new(cc))),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinderaModeSel { Normal /*, Decompose*/ }

struct LabApp {
    // inputs
    query: String,
    lindera_mode: LinderaModeSel,
    ngram_min: String,
    ngram_max: String,
    // runtime
    index: Option<IndexCtx>,
    // outputs
    l_toks: Vec<String>,
    qp_toks: Vec<String>,
    ng_toks: Vec<String>,
    qp_debug: String,
    status: String,
}

struct IndexCtx {
    index: Index,
    field_text: tantivy::schema::Field,
}

impl LabApp {
    fn new(_cc: &CreationContext<'_>) -> Self {
        // Install CJK fallback so Japanese text renders correctly
        install_japanese_fallback_fonts(&_cc.egui_ctx);
        Self {
            query: "肉食の文明の発展".into(),
            lindera_mode: LinderaModeSel::Normal,
            ngram_min: "2".into(),
            ngram_max: "3".into(),
            index: None,
            l_toks: Vec::new(),
            qp_toks: Vec::new(),
            ng_toks: Vec::new(),
            qp_debug: String::new(),
            status: String::from("Ready"),
        }
    }

    fn build_index(&mut self) {
        // Schema with one text field using tokenizer key "ja" (Lindera)
        let mut schema_builder = Schema::builder();
        let mut text_indexing = TextFieldIndexing::default();
        text_indexing = text_indexing.set_tokenizer("ja");
        text_indexing = text_indexing.set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let text_options = TextOptions::default().set_indexing_options(text_indexing);
        let f_text = schema_builder.add_text_field("text", text_options);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema.clone());

        // Register Lindera tokenizer with selected mode
        let dict_uri = dict_uri_from_env();
        let dictionary = load_dictionary(&dict_uri).expect("load lindera dictionary");
        let user_dictionary = None;
        let mode = match self.lindera_mode { LinderaModeSel::Normal => Mode::Normal /*, _ => Mode::Decompose*/ };
        let segmenter = Segmenter::new(mode, dictionary, user_dictionary);
        let tokenizer = LinderaTokenizer::from_segmenter(segmenter);
        index.tokenizers().register("ja", tokenizer);

        self.index = Some(IndexCtx { index, field_text: f_text });
        self.status = "Index initialized (Lindera 'ja')".into();
    }

    fn analyze(&mut self) {
        // Ensure index exists
        if self.index.is_none() { self.build_index(); }
        let Some(ctx) = &self.index else { return; };

        // 1) Lindera tokens via tokenizer directly
        let dict_uri = dict_uri_from_env();
        let dictionary = load_dictionary(&dict_uri).expect("load lindera dictionary");
        let user_dictionary = None;
        let mode = match self.lindera_mode { LinderaModeSel::Normal => Mode::Normal };
        let segmenter = Segmenter::new(mode, dictionary, user_dictionary);
        let mut lindera_tok = LinderaTokenizer::from_segmenter(segmenter);
        let mut stream = lindera_tok.token_stream(self.query.as_str());
        self.l_toks.clear();
        while stream.advance() {
            let t = stream.token();
            self.l_toks.push(t.text.clone());
        }

        // 2) QueryParser side: use analyzer for field
        self.qp_toks.clear();
        if let Ok(mut analyzer) = ctx.index.tokenizer_for_field(ctx.field_text) {
            let mut st = analyzer.token_stream(self.query.as_str());
            while st.advance() {
                self.qp_toks.push(st.token().text.clone());
            }
        }

        // Also parse query and show debug
        let parser = QueryParser::for_index(&ctx.index, vec![ctx.field_text]);
        match parser.parse_query(self.query.as_str()) {
            Ok(q) => self.qp_debug = format!("{:?}", q),
            Err(e) => self.qp_debug = format!("parse error: {e}"),
        }

        // 3) N-gram tokens
        self.ng_toks.clear();
        let min = self.ngram_min.trim().parse::<usize>().unwrap_or(2);
        let max = self.ngram_max.trim().parse::<usize>().unwrap_or(3);
        let mut ng = NgramTokenizer::new(min, max, false)
            .unwrap_or_else(|_| NgramTokenizer::new(2, 3, false).expect("ngram tokenizer"));
        let mut ngs = ng.token_stream(self.query.as_str());
        while ngs.advance() {
            self.ng_toks.push(ngs.token().text.clone());
        }
    }
}

fn dict_uri_from_env() -> String {
    if let Ok(uri) = std::env::var("LINDERA_DICT_URI") { return uri; }
    if let Ok(dir) = std::env::var("LINDERA_DICT_DIR") {
        let p = std::path::PathBuf::from(dir);
        let abs = if p.is_absolute() { p } else { std::env::current_dir().unwrap().join(p) };
        let mut s = abs.to_string_lossy().replace('\\', "/");
        if !s.starts_with('/') { s = format!("/{s}"); }
        return format!("file://{s}");
    }
    // Default to embedded dictionary when features are enabled
    "embedded://ipadic".to_string()
}

impl App for LabApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        CentralPanel::default().show(ctx, |ui| {
            ScrollArea::vertical().id_source("lab_root").show(ui, |ui| {
                ui.heading("Tokenizer Lab (Lindera / QueryParser / N-gram)");
                ui.label(format!("Status: {}", self.status));
                ui.separator();

                ui.horizontal(|ui| {
                    ui.label("Query:");
                    ui.add(TextEdit::singleline(&mut self.query).desired_width(420.0));
                    if ui.add(Button::new("Analyze")).clicked() { self.analyze(); }
                });

                ui.horizontal(|ui| {
                    ui.label("Lindera mode:");
                    let mut mode = self.lindera_mode;
                    ui.selectable_value(&mut mode, LinderaModeSel::Normal, "Normal");
                    // ui.selectable_value(&mut mode, LinderaModeSel::Decompose, "Decompose");
                    if mode != self.lindera_mode { self.lindera_mode = mode; self.build_index(); }
                });

                ui.horizontal(|ui| {
                    ui.label("N-gram min:");
                    ui.add(TextEdit::singleline(&mut self.ngram_min).desired_width(60.0));
                    ui.label("max:");
                    ui.add(TextEdit::singleline(&mut self.ngram_max).desired_width(60.0));
                });

                ui.separator();
                ui.heading("Tokens");
                let header_h = 20.0;
                let row_h = 20.0;
                TableBuilder::new(ui)
                    .column(Column::exact(160.0))
                    .column(Column::remainder())
                    .header(header_h, |mut h| {
                        h.col(|ui| { ui.label("Source"); });
                        h.col(|ui| { ui.label("Tokens"); });
                    })
                    .body(|mut body| {
                        body.row(row_h, |mut r| {
                            r.col(|ui| { ui.label("Lindera"); });
                            r.col(|ui| { ui.monospace(self.l_toks.join(" | ")); });
                        });
                        body.row(row_h, |mut r| {
                            r.col(|ui| { ui.label("QueryParser analyzer"); });
                            r.col(|ui| { ui.monospace(self.qp_toks.join(" | ")); });
                        });
                        body.row(row_h, |mut r| {
                            r.col(|ui| { ui.label("N-gram"); });
                            r.col(|ui| { ui.monospace(self.ng_toks.join(" | ")); });
                        });
                    });

                ui.separator();
                ui.heading("QueryParser Debug");
                ScrollArea::vertical().id_source("qp_dbg").max_height(160.0).show(ui, |ui| {
                    ui.monospace(&self.qp_debug);
                });
            });
        });
    }
}

// --- Japanese font fallback (CJK) -----------------------------
fn install_japanese_fallback_fonts(ctx: &egui::Context) {
    if let Some(data) = load_cjk_font_data() {
        let mut fonts = eframe::egui::FontDefinitions::default();
        fonts
            .font_data
            .insert("jp_fallback".into(), eframe::egui::FontData::from_owned(data));

        for family in [eframe::egui::FontFamily::Proportional, eframe::egui::FontFamily::Monospace] {
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

    // 3) macOS
    for candidate in [
        "/System/Library/Fonts/Hiragino Sans W3.ttc",
        "/System/Library/Fonts/Hiragino Sans W6.ttc",
        "/Library/Fonts/Osaka.ttf",
    ] {
        paths.push(PathBuf::from(candidate));
    }

    // 4) Linux Noto CJK
    for candidate in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ] {
        paths.push(PathBuf::from(candidate));
    }

    // 5) Project-local fallback
    paths.push(PathBuf::from("fonts/NotoSansJP-Regular.otf"));

    paths
}
