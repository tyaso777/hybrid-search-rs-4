use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::{mpsc::{self, Receiver, TryRecvError}, Arc};
use std::time::Instant;

use eframe::egui::{self, Button, CentralPanel, ComboBox, ScrollArea, Spinner, TextEdit};
use eframe::egui::ProgressBar;
use egui_extras::{Column, TableBuilder};
use eframe::{App, CreationContext, Frame, NativeOptions};
use rfd::FileDialog;

use hybrid_service::{HybridService, ServiceConfig, CancelToken, ProgressEvent, HnswState};
use embedding_provider::config::{default_stdio_config, ONNX_STDIO_DEFAULTS};
use chunking_store::FilterClause;
use chunking_store::ChunkStoreRead;
use chunking_store::{FilterKind, FilterOp};
#[cfg(feature = "tantivy")]
use chunking_store::tantivy_index::{TantivyIndex, TokenCombine};
use chunk_model::{ChunkId, DocumentId, ChunkRecord, SCHEMA_MAJOR};

fn main() -> eframe::Result<()> {
    let options = NativeOptions::default();
    eframe::run_native(
        "Hybrid Service GUI",
        options,
        Box::new(|cc| Box::new(AppState::new(cc))),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveTab {
    Insert,
    Search,
    Config,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InsertMode {
    File,
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchMode {
    Hybrid,
    Tantivy,
    Vec,
}

#[derive(Debug)]
struct ServiceInitTask {
    rx: Receiver<Result<Arc<HybridService>, String>>,
    started: Instant,
}

#[derive(Debug, Clone, Default)]
struct HitRow {
    cid: String,
    file: String,
    page: String,
    text_preview: String,
    text_full: String,
    tv: Option<f32>,
    tv_and: Option<f32>,
    tv_or: Option<f32>,
    vec: Option<f32>,
}

struct AppState {
    // Model config
    model_path: String,
    tokenizer_path: String,
    runtime_path: String,
    embedding_dimension: String,
    max_tokens: String,
    embed_batch_size: String,
    embed_auto: bool,
    embed_initial_batch: String,
    embed_min_batch: String,
    preload_model_to_memory: bool,

    // Store/index config (root -> derive artifacts)
    store_root: String,
    db_path: String,
    hnsw_dir: String,
    #[cfg(feature = "tantivy")]
    tantivy_dir: String,
    #[cfg(feature = "tantivy")]
    tantivy: Option<TantivyIndex>,

    // Service
    svc: Option<Arc<HybridService>>,
    svc_task: Option<ServiceInitTask>,

    // Insert
    input_text: String,
    doc_hint: String,
    ingest_file_path: String,
    ingest_encoding: String,
    ingest_preview: String,

    // Chunk params (unified for PDF/TXT)
    chunk_min: String,
    chunk_max: String,
    chunk_cap: String,
    chunk_penalize_short_line: bool,
    chunk_penalize_page_no_nl: bool,

    // Ingest job (async)
    ingest_rx: Option<Receiver<ProgressEvent>>,
    ingest_cancel: Option<CancelToken>,
    ingest_running: bool,
    ingest_done: usize,
    ingest_total: usize,
    ingest_last_batch: usize,
    ingest_started: Option<Instant>,
    ingest_doc_key: Option<String>,

    // Search
    query: String,
    top_k: usize,
    w_text: f32,
    w_vec: f32,
    results: Vec<HitRow>,
    search_mode: SearchMode,

    // UI
    tab: ActiveTab,
    insert_mode: InsertMode,
    status: String,
    selected_cid: Option<String>,
    selected_text: String,
    selected_display: String,
}

impl AppState {
    fn ui_config(&mut self, ui: &mut egui::Ui) {
        ui.heading("Model / Store Config");
        ui.add_enabled_ui(!self.ingest_running, |ui| {
            ui.horizontal(|ui| {
                ui.label("Store Root");
                if ui.button("Browse").clicked() {
                    if let Some(p) = FileDialog::new().pick_folder() {
                        self.store_root = p.display().to_string();
                        self.refresh_store_paths();
                        let _ = fs::create_dir_all(self.store_root.trim());
                        let _ = fs::create_dir_all(derive_hnsw_dir(self.store_root.trim()));
                        #[cfg(feature = "tantivy")] let _ = fs::create_dir_all(derive_tantivy_dir(self.store_root.trim()));
                        self.status = format!("Store root set to {}", self.store_root.trim()); if let Some(svc) = &self.svc { svc.set_store_paths(PathBuf::from(self.db_path.trim()), Some(PathBuf::from(self.hnsw_dir.trim()))); }
                    }
                }
                ui.add(TextEdit::singleline(&mut self.store_root).desired_width(400.0));
                if ui.button("Set").clicked() {
                    self.refresh_store_paths();
                    let _ = fs::create_dir_all(self.store_root.trim());
                    let _ = fs::create_dir_all(derive_hnsw_dir(self.store_root.trim()));
                    #[cfg(feature = "tantivy")]
                    let _ = fs::create_dir_all(derive_tantivy_dir(self.store_root.trim()));
                    self.status = format!("Store root set to {}", self.store_root.trim()); if let Some(svc) = &self.svc { svc.set_store_paths(PathBuf::from(self.db_path.trim()), Some(PathBuf::from(self.hnsw_dir.trim()))); }
                }
                if ui.button("Reset").clicked() {
                    self.store_root = "target/demo/store".into();
                    self.refresh_store_paths();
                    let _ = fs::create_dir_all(self.store_root.trim());
                    let _ = fs::create_dir_all(derive_hnsw_dir(self.store_root.trim()));
                    #[cfg(feature = "tantivy")]
                    let _ = fs::create_dir_all(derive_tantivy_dir(self.store_root.trim()));
                    self.status = format!("Store root reset to {}", self.store_root.trim());
                }
            });
            ui.horizontal(|ui| { ui.label("DB"); ui.label(&self.db_path); });
            ui.horizontal(|ui| { ui.label("HNSW"); ui.label(&self.hnsw_dir); });
            #[cfg(feature = "tantivy")]
            ui.horizontal(|ui| { ui.label("Tantivy"); ui.label(&self.tantivy_dir); });
            ui.separator();
            // Chunking Params (always visible)
            ui.horizontal(|ui| {
                ui.label("Chunking Params");
                ui.label("min"); ui.add(TextEdit::singleline(&mut self.chunk_min).desired_width(60.0));
                ui.label("max"); ui.add(TextEdit::singleline(&mut self.chunk_max).desired_width(60.0));
                ui.label("cap"); ui.add(TextEdit::singleline(&mut self.chunk_cap).desired_width(60.0));
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.chunk_penalize_short_line, "Penalize after short line");
                ui.checkbox(&mut self.chunk_penalize_page_no_nl, "Penalize page-boundary without newline");
            });
            ui.collapsing("Danger zone", |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Deletes DB and indexes (HNSW/Tantivy)").color(egui::Color32::LIGHT_RED));
                    let btn = egui::RichText::new("Delete DB & Indexes").color(egui::Color32::RED);
                    if ui.button(btn).clicked() { self.delete_store_files(); }
                });
            });
            ui.separator();
            ui.horizontal(|ui| { ui.label("Model"); ui.add(TextEdit::singleline(&mut self.model_path).desired_width(400.0)); if ui.button("Browse").clicked() { if let Some(p) = FileDialog::new().add_filter("ONNX", &["onnx"]).pick_file() { self.model_path = p.display().to_string(); } } });
            ui.horizontal(|ui| { ui.label("Tokenizer"); ui.add(TextEdit::singleline(&mut self.tokenizer_path).desired_width(400.0)); if ui.button("Browse").clicked() { if let Some(p) = FileDialog::new().add_filter("JSON", &["json"]).pick_file() { self.tokenizer_path = p.display().to_string(); } } });
            ui.horizontal(|ui| { ui.label("Runtime DLL"); ui.add(TextEdit::singleline(&mut self.runtime_path).desired_width(400.0)); if ui.button("Browse").clicked() { if let Some(p) = FileDialog::new().pick_file() { self.runtime_path = p.display().to_string(); } } });
            ui.horizontal(|ui| {
                ui.label("Dim"); ui.add(TextEdit::singleline(&mut self.embedding_dimension).desired_width(80.0));
                ui.label("MaxTokens"); ui.add(TextEdit::singleline(&mut self.max_tokens).desired_width(80.0));
                ui.label("Batch"); ui.add(TextEdit::singleline(&mut self.embed_batch_size).desired_width(60.0));
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.preload_model_to_memory, "Preload model into memory");
                ui.checkbox(&mut self.embed_auto, "Auto batch");
            });
            ui.collapsing("Auto batch settings", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Initial"); ui.add(TextEdit::singleline(&mut self.embed_initial_batch).desired_width(60.0));
                    ui.label("Min"); ui.add(TextEdit::singleline(&mut self.embed_min_batch).desired_width(60.0));
                });
            });
        });
    }
    fn refresh_store_paths(&mut self) {
        let root = self.store_root.trim();
        self.db_path = derive_db_path(root);
        self.hnsw_dir = derive_hnsw_dir(root);
        #[cfg(feature = "tantivy")]
        { self.tantivy_dir = derive_tantivy_dir(root); }
    }

    #[cfg(feature = "tantivy")]
    fn ensure_tantivy_open(&mut self) -> Result<(), String> {
        if self.tantivy.is_none() {
            let dir = &self.tantivy_dir;
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            let idx = TantivyIndex::open_or_create_dir(dir).map_err(|e| e.to_string())?;
            self.tantivy = Some(idx);
        }
        Ok(())
    }

    fn delete_store_files(&mut self) {
        // Close any open service to release SQLite handles before deleting
        self.svc = None;
        // Delete DB and sidecar files (-wal, -shm)
        let db = std::path::PathBuf::from(self.db_path.trim());
        let mut removed: Vec<String> = Vec::new();
        let mut errs: Vec<String> = Vec::new();
        let mut try_remove = |p: &std::path::Path| {
            if p.exists() {
                match std::fs::remove_file(p) {
                    Ok(_) => removed.push(p.display().to_string()),
                    Err(e) => errs.push(format!("{}: {}", p.display(), e)),
                }
            }
        };
        try_remove(&db);
        // Remove SQLite sidecar files (WAL/SHM)
        if let Some(s) = db.to_str() { try_remove(std::path::Path::new(&format!("{}-wal", s))); try_remove(std::path::Path::new(&format!("{}-shm", s))); }
        // HNSW dir
        let hdir = std::path::PathBuf::from(self.hnsw_dir.trim());
        if hdir.exists() {
            match std::fs::remove_dir_all(&hdir) { Ok(_) => removed.push(hdir.display().to_string()), Err(e) => errs.push(format!("{}: {}", hdir.display(), e)) }
        }
        // Tantivy dir
        #[cfg(feature = "tantivy")]
        {
            let tdir = std::path::PathBuf::from(self.tantivy_dir.trim());
            if tdir.exists() {
                match std::fs::remove_dir_all(&tdir) { Ok(_) => removed.push(tdir.display().to_string()), Err(e) => errs.push(format!("{}: {}", tdir.display(), e)) }
            }
        }

        if errs.is_empty() {
            if removed.is_empty() {
                self.status = "Delete: nothing to remove".into();
            } else {
                self.status = format!("Deleted: {}", removed.join(", "));
            }
        } else {
            self.status = format!("Deleted: {}  Errors: {}",
                if removed.is_empty() { String::from("<none>") } else { removed.join(", ") },
                errs.join(", ")
            );
        }
    }
    fn new(cc: &CreationContext<'_>) -> Self {
        install_japanese_fallback_fonts(&cc.egui_ctx);
        let defaults = default_stdio_config();
        let store_default = String::from("target/demo/store");
        Self {
            model_path: defaults.model_path.display().to_string(),
            tokenizer_path: defaults.tokenizer_path.display().to_string(),
            runtime_path: defaults.runtime_library_path.display().to_string(),
            embedding_dimension: ONNX_STDIO_DEFAULTS.embedding_dimension.to_string(),
            max_tokens: ONNX_STDIO_DEFAULTS.max_input_tokens.to_string(),
            embed_batch_size: String::from("64"),
            embed_auto: true,
            embed_initial_batch: String::from("128"),
            embed_min_batch: String::from("8"),
            preload_model_to_memory: false,

            store_root: store_default.clone(),
            db_path: derive_db_path(&store_default),
            hnsw_dir: derive_hnsw_dir(&store_default),
            #[cfg(feature = "tantivy")]
            tantivy_dir: derive_tantivy_dir(&store_default),
            #[cfg(feature = "tantivy")]
            tantivy: None,

            svc: None,
            svc_task: None,

            input_text: String::new(),
            doc_hint: String::new(),
            ingest_file_path: String::new(),
            ingest_encoding: String::from("auto"),
            ingest_preview: String::new(),

            chunk_min: String::from("400"),
            chunk_max: String::from("600"),
            chunk_cap: String::from("800"),
            chunk_penalize_short_line: true,
            chunk_penalize_page_no_nl: true,

            ingest_rx: None,
            ingest_cancel: None,
            ingest_running: false,
            ingest_done: 0,
            ingest_total: 0,
            ingest_last_batch: 0,
            ingest_started: None,
            ingest_doc_key: None,

            query: String::new(),
            top_k: 10,
            w_text: 0.5,
            w_vec: 0.5,
            results: Vec::new(),
            search_mode: SearchMode::Hybrid,

            tab: ActiveTab::Insert,
            insert_mode: InsertMode::File,
            status: String::new(),
            selected_cid: None,
            selected_text: String::new(),
            selected_display: String::new(),
        }
    }

    fn model_not_initialized(&self) -> bool { self.svc.is_none() && self.svc_task.is_none() }

    fn release_model_and_indexes(&mut self) {
        // Drop service (ONNX session + resident HNSW) and Tantivy handle if any
        self.svc = None;
        #[cfg(feature = "tantivy")] {
            self.tantivy = None;
        }
        self.status = "Released model and resident indexes".into();
    }

    fn start_service_init(&mut self) {
        // Build config from UI fields
        let root = self.store_root.trim().to_string();
        let db = derive_db_path(&root);
        let hnsw = derive_hnsw_dir(&root);
        // ensure directories
        let _ = fs::create_dir_all(&root);
        let _ = fs::create_dir_all(&hnsw);
        #[cfg(feature = "tantivy")]
        let _ = fs::create_dir_all(derive_tantivy_dir(&root));
        let mut cfg = ServiceConfig::default();
        cfg.db_path = PathBuf::from(db);
        cfg.hnsw_dir = Some(PathBuf::from(hnsw));
        cfg.embedder.model_path = PathBuf::from(self.model_path.trim());
        cfg.embedder.tokenizer_path = PathBuf::from(self.tokenizer_path.trim());
        cfg.embedder.runtime_library_path = PathBuf::from(self.runtime_path.trim());
        cfg.embedder.dimension = self.embedding_dimension.trim().parse().unwrap_or(ONNX_STDIO_DEFAULTS.embedding_dimension);
        cfg.embedder.max_input_length = self.max_tokens.trim().parse().unwrap_or(ONNX_STDIO_DEFAULTS.max_input_tokens);
        cfg.embedder.preload_model_to_memory = self.preload_model_to_memory;
        // Embed batch size
        if let Ok(bs) = self.embed_batch_size.trim().parse::<usize>() { if bs > 0 { cfg.embed_batch_size = bs; } }
        // Auto batch params
        cfg.embed_auto = self.embed_auto;
        if let Ok(x) = self.embed_initial_batch.trim().parse::<usize>() { if x > 0 { cfg.embed_initial_batch = x; } }
        if let Ok(x) = self.embed_min_batch.trim().parse::<usize>() { if x > 0 { cfg.embed_min_batch = x; } }

        let (tx, rx) = mpsc::channel();
        self.status = "Initializing model...".into();
        self.svc_task = Some(ServiceInitTask { rx, started: Instant::now() });
        std::thread::spawn(move || {
            let res = HybridService::new(cfg).map(|s| Arc::new(s)).map_err(|e| e.to_string());
            let _ = tx.send(res);
        });
    }

    fn poll_service_task(&mut self) {
        if let Some(task) = &self.svc_task {
            match task.rx.try_recv() {
                Ok(Ok(svc)) => {
                    self.svc = Some(svc);
                    self.status = format!("Model ready in {:.1}s", task.started.elapsed().as_secs_f32());
                    self.svc_task = None;
                }
                Ok(Err(err)) => {
                    self.status = format!("Init failed: {err}");
                    self.svc_task = None;
                }
                Err(TryRecvError::Empty) => {
                    // still working
                }
                Err(TryRecvError::Disconnected) => {
                    self.status = "Init failed: worker disconnected".into();
                    self.svc_task = None;
                }
            }
        }
    }
}

impl App for AppState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        self.poll_service_task();
        self.poll_ingest_job();
        CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add_enabled_ui(!self.ingest_running, |ui| {
                    ui.selectable_value(&mut self.tab, ActiveTab::Insert, "Insert");
                    ui.selectable_value(&mut self.tab, ActiveTab::Search, "Search");
                    ui.selectable_value(&mut self.tab, ActiveTab::Config, "Config");
                    // stronger visual divider (double vertical separator)
                    ui.separator();
                    ui.separator();
                    let (model_status, index_status) = if self.svc.is_none() {
                        if self.svc_task.is_some() { ("loading", "absent") } else { ("released", "absent") }
                    } else {
                        let st = self.svc.as_ref().map(|s| s.hnsw_state()).unwrap_or(HnswState::Error);
                        let idx = match st { HnswState::Absent => "absent", HnswState::Loading => "loading", HnswState::Ready => "ready", HnswState::Error => "error" };
                        ("ready", idx)
                    };
                    let status_text = format!("Model: {}, Index: {}", model_status, index_status);
                    ui.label(status_text);
                    // compact controls with a clear right-side divider as well
                    ui.add_space(8.0);
                    ui.add_space(8.0);
                    if ui.button("Init").clicked() {
                        if self.model_not_initialized() { self.start_service_init(); }
                    }
                    if ui.button("Release").clicked() {
                        self.release_model_and_indexes();
                    }
                    // trailing divider (double) to close the group
                    ui.separator();
                    ui.separator();
                });
                if self.ingest_running {
                    ui.add(Spinner::new());
                    if ui.add(Button::new("Cancel")).clicked() {
                        if let Some(ct) = &self.ingest_cancel { ct.cancel(); }
                    }
                }
            });

            ui.separator();
            match self.tab {
                ActiveTab::Insert => self.ui_insert(ui),
                ActiveTab::Search => self.ui_search(ui),
                ActiveTab::Config => self.ui_config(ui),
            }

            ui.separator();
            if !self.status.is_empty() { ui.label(&self.status); }
        });
    }
}

impl AppState {
    fn ui_insert(&mut self, ui: &mut egui::Ui) {
        ui.heading("Insert");
        ui.add_enabled_ui(!self.ingest_running, |ui| {
            // Sub-tabs for Insert
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.insert_mode, InsertMode::File, "Insert File");
                ui.selectable_value(&mut self.insert_mode, InsertMode::Text, "Insert Text");
            });

            match self.insert_mode {
                InsertMode::File => {
                    // Row 1: Choose File, [File path], Ingest File
                    ui.horizontal(|ui| {
                        if ui.button("Choose File").clicked() {
                            if let Some(p) = FileDialog::new().pick_file() {
                                self.ingest_file_path = p.display().to_string();
                                // Reset encoding to auto and refresh preview when a new file is chosen
                                self.ingest_encoding = String::from("auto");
                                self.refresh_ingest_preview();
                            }
                        }
                        let resp = ui.add(TextEdit::singleline(&mut self.ingest_file_path).desired_width(400.0));
                        if resp.changed() { self.refresh_ingest_preview(); }
                        let ingest_btn = ui.add_enabled(!self.ingest_running, Button::new("Ingest File"));
                        if ingest_btn.clicked() { self.do_ingest_file(); }
                    });

                    // Row 2: Encoding selector (for text-like files) and short preview
                    if self.is_text_like_path(self.ingest_file_path.trim()) {
                        ui.horizontal(|ui| {
                            ui.label("Encoding");
                            let encs = ["auto", "utf-8", "shift_jis", "windows-1252", "utf-16le", "utf-16be"];
                            let before = self.ingest_encoding.clone();
                            ComboBox::from_id_source("ingest_encoding_combo")
                                .selected_text(self.ingest_encoding.clone())
                                .show_ui(ui, |ui| {
                                    for e in encs { ui.selectable_value(&mut self.ingest_encoding, e.to_string(), e); }
                                });
                            if self.ingest_encoding != before { self.refresh_ingest_preview(); }
                            let mut preview_short: String = self.ingest_preview.chars().take(60).collect();
                            if self.ingest_preview.chars().count() > 60 { preview_short.push('…'); }
                            ui.label(format!("Preview: {}", preview_short.replace(['\n','\r','\t'], " ")));
                        });
                    }
                }
                InsertMode::Text => {
                    // Text input with the action button placed below
                    ui.label("Text");
                    ui.add(TextEdit::multiline(&mut self.input_text).desired_rows(4).desired_width(600.0));
                    if ui.add(Button::new("Insert Text")).clicked() { self.do_insert_text(); }
                }
            }
        });
        if self.ingest_running {
            ui.horizontal(|ui| {
                let frac = if self.ingest_total > 0 { (self.ingest_done as f32 / self.ingest_total as f32).clamp(0.0, 1.0) } else { 0.0 };
                ui.add(ProgressBar::new(frac).desired_width(400.0).show_percentage());
                ui.label(format!("{} / {} (batch {})", self.ingest_done, self.ingest_total, self.ingest_last_batch));
                if ui.add(Button::new("Cancel")).clicked() {
                    if let Some(ct) = &self.ingest_cancel { ct.cancel(); }
                }
                if let Some(started) = self.ingest_started {
                    let secs = started.elapsed().as_secs_f32();
                    ui.label(format!("{:.1}s", secs));
                }
            });
        }
    }

    fn is_text_like_path(&self, path: &str) -> bool {
        let lower = path.to_ascii_lowercase();
        let exts = [
            ".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".json", ".yaml", ".yml",
            ".ini", ".toml", ".cfg", ".conf", ".rst", ".tex", ".srt", ".properties",
        ];
        exts.iter().any(|e| lower.ends_with(e))
    }

    fn refresh_ingest_preview(&mut self) {
        use std::fs;
        let path = self.ingest_file_path.trim();
        self.ingest_preview.clear();
        if path.is_empty() { return; }
        let Ok(bytes) = fs::read(path) else { return; };
        let enc = self.ingest_encoding.to_ascii_lowercase();
        let mut text = match enc.as_str() {
            "auto" => String::from_utf8_lossy(&bytes).to_string(),
            "utf-8" | "utf8" => String::from_utf8_lossy(&bytes).to_string(),
            "shift_jis" | "sjis" | "cp932" | "windows-31j" => {
                let (cow, _, _) = encoding_rs::SHIFT_JIS.decode(&bytes); cow.into_owned()
            }
            "windows-1252" | "cp1252" => { let (cow, _, _) = encoding_rs::WINDOWS_1252.decode(&bytes); cow.into_owned() }
            "utf-16le" | "utf16le" => {
                let mut u16s: Vec<u16> = Vec::with_capacity(bytes.len()/2);
                let mut i = 0usize; if bytes.len() >= 2 && bytes[0]==0xFF && bytes[1]==0xFE { i = 2; }
                while i + 1 < bytes.len() { u16s.push(u16::from_le_bytes([bytes[i], bytes[i+1]])); i += 2; }
                String::from_utf16_lossy(&u16s)
            }
            "utf-16be" | "utf16be" => {
                let mut u16s: Vec<u16> = Vec::with_capacity(bytes.len()/2);
                let mut i = 0usize; if bytes.len() >= 2 && bytes[0]==0xFE && bytes[1]==0xFF { i = 2; }
                while i + 1 < bytes.len() { u16s.push(u16::from_be_bytes([bytes[i], bytes[i+1]])); i += 2; }
                String::from_utf16_lossy(&u16s)
            }
            _ => String::from_utf8_lossy(&bytes).to_string(),
        };
        // normalize CRLF
        text = text.replace('\r', "");
        self.ingest_preview = text;
    }

    fn do_insert_text(&mut self) {
        let text = self.input_text.trim().to_string();
        if text.is_empty() { self.status = "Enter some text to insert".into(); return; }
        let Some(svc) = self.svc.as_ref() else { self.status = "Model not initialized".into(); return; };
        match svc.ingest_text(&text, if self.doc_hint.trim().is_empty() { None } else { Some(self.doc_hint.trim()) }) {
            Ok((doc_id, chunk_id)) => {
                // Upsert into Tantivy (optional)
                #[cfg(feature = "tantivy")]
                {
                    if let Err(err) = self.ensure_tantivy_open() { self.status = format!("Tantivy init failed: {err}"); return; }
                    if let Some(idx) = &self.tantivy {
                        let rec = rec_clone_for_tantivy(&doc_id, &chunk_id, &text);
                        if let Err(e) = idx.upsert_records(&[rec]) { self.status = format!("Tantivy upsert failed: {e}"); return; }
                    }
                }
                self.status = format!("Inserted chunk {} (doc={})", chunk_id.0, doc_id.0);
            }
            Err(e) => { self.status = format!("Insert failed: {e}"); }
        }
    }

    fn do_ingest_file(&mut self) {
        let path_owned = self.ingest_file_path.trim().to_string();
        if path_owned.is_empty() { self.status = "Choose a file to ingest".into(); return; }
        let Some(svc) = self.svc.as_ref() else { self.status = "Model not initialized".into(); return; };
        let svc = Arc::clone(svc);
        let doc_hint_opt = if self.doc_hint.trim().is_empty() { None } else { Some(self.doc_hint.trim().to_string()) };
        let (tx, rx) = mpsc::channel::<ProgressEvent>();
        let cancel = CancelToken::new();
        self.ingest_rx = Some(rx);
        self.ingest_cancel = Some(cancel.clone());
        self.ingest_running = true;
        self.ingest_done = 0;
        self.ingest_total = 0;
        self.ingest_last_batch = 0;
        self.ingest_started = Some(Instant::now());
        // Remember the doc key used for Tantivy upsert after finish
        self.ingest_doc_key = Some(if self.doc_hint.trim().is_empty() { path_owned.clone() } else { self.doc_hint.trim().to_string() });
        self.status = format!("Ingesting file: {}", path_owned);
        let enc_opt = { let e = self.ingest_encoding.trim().to_string(); if e.is_empty() || e.eq_ignore_ascii_case("auto") { None } else { Some(e) } };
        let min = self.chunk_min.trim().parse().unwrap_or(400);
        let max = self.chunk_max.trim().parse().unwrap_or(600);
        let cap = self.chunk_cap.trim().parse().unwrap_or(800);
        let ps = self.chunk_penalize_short_line;
        let pp = self.chunk_penalize_page_no_nl;
        std::thread::spawn(move || {
            let hint = doc_hint_opt.as_deref();
            let cb: Box<dyn FnMut(ProgressEvent) + Send> = Box::new(move |ev: ProgressEvent| { let _ = tx.send(ev); });
            let _ = svc.ingest_file_with_progress_custom(
                &path_owned, hint,
                enc_opt.as_deref(),
                min, max, cap, ps, pp,
                Some(&cancel), Some(cb)
            );
            // The service emits Finished/Canceled; no-op here.
        });
    }

    fn poll_ingest_job(&mut self) {
        if let Some(rx) = &self.ingest_rx {
            loop {
                match rx.try_recv() {
                    Ok(ev) => {
                        match ev {
                            ProgressEvent::Start { total_chunks } => {
                                self.ingest_total = total_chunks;
                                self.ingest_done = 0;
                                self.ingest_last_batch = 0;
                                if self.ingest_started.is_none() { self.ingest_started = Some(Instant::now()); }
                                self.status = format!("Embedding {} chunks...", total_chunks);
                            }
                            ProgressEvent::EmbedBatch { done, total, batch } => {
                                self.ingest_done = done;
                                self.ingest_total = total;
                                self.ingest_last_batch = batch;
                                self.status = format!("Embedding: {} / {} (batch {})", done, total, batch);
                            }
                            ProgressEvent::UpsertDb { total } => {
                                self.status = format!("Upserting into DB ({} chunks)...", total);
                            }
                            ProgressEvent::IndexText { total } => {
                                self.status = format!("Indexing text ({} chunks)...", total);
                            }
                            ProgressEvent::IndexVector { total } => {
                                self.status = format!("Indexing vectors ({} chunks)...", total);
                            }
                            ProgressEvent::SaveIndexes => {
                                self.status = "Saving indexes...".into();
                            }
                            ProgressEvent::Finished { total } => {
                                self.ingest_running = false;
                                self.ingest_cancel = None;
                                self.ingest_rx = None;
                                let secs = self.ingest_started.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
                                self.status = format!("Ingest finished ({} chunks) in {:.1}s.", total, secs);
                                self.ingest_started = None;
                                // After ingest, upsert into Tantivy (if enabled)
                                #[cfg(feature = "tantivy")]
                                {
                                    if let Some(doc_key) = self.ingest_doc_key.clone() {
                                        if let Err(err) = self.ensure_tantivy_open() { self.status = format!("Tantivy init failed: {err}"); break; }
                                        use chunking_store::sqlite_repo::SqliteRepo;
                                        let repo = match SqliteRepo::open(self.db_path.trim()) { Ok(r) => r, Err(e) => { self.status = format!("Open DB failed: {e}"); break; } };
                                        let filter = FilterClause { kind: FilterKind::Must, op: FilterOp::DocIdEq(doc_key) };
                                        let ids = match repo.list_chunk_ids_by_filter(&[filter], 50_000, 0) { Ok(v) => v, Err(_) => Vec::new() };
                                        if !ids.is_empty() {
                                            if let Ok(recs) = repo.get_chunks_by_ids(&ids) {
                                                if let Some(idx) = &self.tantivy { let _ = idx.upsert_records(&recs); }
                                            }
                                        }
                                    }
                                    self.ingest_doc_key = None;
                                }
                                break;
                            }
                            ProgressEvent::Canceled => {
                                self.ingest_running = false;
                                self.ingest_cancel = None;
                                self.ingest_rx = None;
                                let secs = self.ingest_started.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
                                self.status = format!("Ingest canceled after {:.1}s.", secs);
                                self.ingest_started = None;
                                break;
                            }
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.ingest_running = false;
                        self.ingest_cancel = None;
                        self.ingest_rx = None;
                        self.status = "Ingest worker disconnected".into();
                        break;
                    }
                }
            }
        }
    }

    fn ui_search(&mut self, ui: &mut egui::Ui) {
        ui.push_id("search_panel", |ui| {
            ui.add_enabled_ui(!self.ingest_running, |ui| {
                ui.heading("Search");
                // Row 1: Query + Search
                ui.horizontal(|ui| {
                    ui.label("Query");
                    ui.add(TextEdit::singleline(&mut self.query).desired_width(400.0).id_source("search_query"));
                    if ui.add(Button::new("Search")).clicked() { self.do_search_now(); }
                });
            // Row 2: Options (TopK / Mode slider / Weights when Hybrid)
            ui.horizontal(|ui| {
                ui.label("TopK");
                let mut topk_str = self.top_k.to_string();
                if ui.add(TextEdit::singleline(&mut topk_str).desired_width(60.0).id_source("search_topk")).changed() {
                    self.top_k = topk_str.parse().unwrap_or(10);
                }
                ui.separator();
                ui.label("Mode");
                let mut idx = match self.search_mode { SearchMode::Hybrid => 0, SearchMode::Tantivy => 1, SearchMode::Vec => 2 };
                if ui.add(egui::Slider::new(&mut idx, 0..=2).show_value(false).clamp_to_range(true).smart_aim(false)).changed() {
                    self.search_mode = match idx { 1 => SearchMode::Tantivy, 2 => SearchMode::Vec, _ => SearchMode::Hybrid };
                }
                let mode_name = match self.search_mode { SearchMode::Hybrid => "Hybrid", SearchMode::Tantivy => "Tantivy", SearchMode::Vec => "VEC" };
                ui.label(mode_name);
                if matches!(self.search_mode, SearchMode::Hybrid) {
                    ui.separator();
                    ui.label("w_TV(OR)");
                    ui.add(egui::DragValue::new(&mut self.w_text).speed(0.1).clamp_range(0.0..=100.0));
                    ui.label("w_VEC");
                    ui.add(egui::DragValue::new(&mut self.w_vec).speed(0.1).clamp_range(0.0..=100.0));
                    let denom = (self.w_text + self.w_vec).max(1e-6);
                    let wt = self.w_text / denom;
                    let wv = self.w_vec / denom;
                    ui.label(format!("{:.0}%/{:.0}%", wt*100.0, wv*100.0));
                }
            });

            ui.separator();
            ui.push_id("results_table", |ui| {
                TableBuilder::new(ui)
            .striped(true)
            .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
            .column(Column::initial(36.0).at_least(30.0))
            .column(Column::initial(220.0))   // file
            .column(Column::initial(80.0))    // page
            .column(Column::remainder())      // text
            .column(Column::initial(70.0))    // TV
            .column(Column::initial(80.0))    // TV(AND)
            .column(Column::initial(80.0))    // TV(OR)
            .column(Column::initial(70.0))    // VEC
            .header(20.0, |mut header| {
                header.col(|ui| { ui.label("#"); });
                header.col(|ui| { ui.label("File"); });
                header.col(|ui| { ui.label("Page"); });
                header.col(|ui| { ui.label("Text"); });
                header.col(|ui| { ui.label("TV"); });
                header.col(|ui| { ui.label("TV(AND)"); });
                header.col(|ui| { ui.label("TV(OR)"); });
                header.col(|ui| { ui.label("VEC"); });
            })
            .body(|mut body| {
                for (i, row) in self.results.iter().enumerate() {
                    body.row(20.0, |mut row_ui| {
                        row_ui.col(|ui| { ui.label(format!("{}", i+1)); });
                        // Make file and page clickable to select the row
                        row_ui.col(|ui| {
                            if ui.link(&row.file).clicked() {
                                self.selected_cid = Some(row.cid.clone());
                                self.selected_text = row.text_full.clone();
                                self.selected_display = format!("{} {}", &row.file, if row.page.is_empty() { String::new() } else { row.page.clone() });
                            }
                        });
                        row_ui.col(|ui| {
                            if !row.page.is_empty() {
                                if ui.link(&row.page).clicked() {
                                    self.selected_cid = Some(row.cid.clone());
                                    self.selected_text = row.text_full.clone();
                                    self.selected_display = format!("{} {}", &row.file, &row.page);
                                }
                            } else {
                                ui.label(&row.page);
                            }
                        });
                        row_ui.col(|ui| {
                            ui.push_id(i, |ui| {
                                // Clickable preview (remove chunk_id link display)
                                if ui.link(&row.text_preview).clicked() {
                                    self.selected_cid = Some(row.cid.clone());
                                    self.selected_text = row.text_full.clone();
                                    self.selected_display = format!("{} {}", &row.file, if row.page.is_empty() { String::new() } else { row.page.clone() });
                                }
                            });
                        });
                        row_ui.col(|ui| { ui.label(opt_fmt(row.tv)); });
                        row_ui.col(|ui| { ui.label(opt_fmt(row.tv_and)); });
                        row_ui.col(|ui| { ui.label(opt_fmt(row.tv_or)); });
                        row_ui.col(|ui| { ui.label(opt_fmt(row.vec)); });
                    });
                }
            });
            });
            });

            if let Some(_cid) = &self.selected_cid {
                ui.separator();
                // Prefer human-friendly Selected header
                if self.selected_display.is_empty() {
                    ui.label("Selected:");
                } else {
                    ui.label(format!("Selected: {}", self.selected_display));
                }
                ScrollArea::vertical().max_height(200.0).id_source("selected_scroll").show(ui, |ui| {
                    ui.add(TextEdit::multiline(&mut self.selected_text).desired_rows(8).desired_width(800.0).id_source("selected_text"));
                });
            }
        });
    }

    fn do_search_now(&mut self) {
        let q_owned = self.query.clone();
        let q = q_owned.trim();
        if q.is_empty() { self.status = "Enter query".into(); return; }
        let filters: &[FilterClause] = &[];
        let top_k = self.top_k;
        // Allow FTS-only search without model initialization
        // Build union of results from TV, TV(AND), TV(OR), VEC
        use chunking_store::sqlite_repo::SqliteRepo;
        use chunking_store::SearchOptions;
        let repo = match SqliteRepo::open(self.db_path.trim()) { Ok(r) => r, Err(e) => { self.status = format!("Open DB failed: {e}"); return; } };
        // FTS5 is not used for search ranking here; skip any FTS maintenance.
        let opts = SearchOptions { top_k, fetch_factor: 10 };

        // Tantivy queries (skip when mode = VEC)
        #[cfg(feature = "tantivy")]
        let (tv, tv_and, tv_or) = if matches!(self.search_mode, SearchMode::Vec) {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            if let Err(err) = self.ensure_tantivy_open() { self.status = format!("Tantivy init failed: {err}"); return; }
            if let Some(ti) = &self.tantivy {
                let a = chunking_store::TextSearcher::search_ids(ti, &repo, q, filters, &opts);
                let b = ti.search_ids_tokenized(&repo, q, filters, &opts, TokenCombine::AND);
                let c = ti.search_ids_tokenized(&repo, q, filters, &opts, TokenCombine::OR);
                (a, b, c)
            } else { (Vec::new(), Vec::new(), Vec::new()) }
        };
        #[cfg(not(feature = "tantivy"))]
        let (tv, tv_and, tv_or) = (Vec::new(), Vec::new(), Vec::new());

        // Vector query (optional): use service.search_hybrid with w_text=0 for vector-only scoring
        let vec_matches: Vec<chunking_store::TextMatch> = if matches!(self.search_mode, SearchMode::Tantivy) {
            Vec::new()
        } else {
            if let Some(svc) = &self.svc {
                match svc.search_hybrid(q, top_k, filters, 0.0, 1.0) {
                    Ok(hits) => hits.into_iter().map(|h| chunking_store::TextMatch { chunk_id: chunk_model::ChunkId(h.chunk.chunk_id.0), score: h.score, raw_score: h.score }).collect(),
                    Err(_) => Vec::new(),
                }
            } else { Vec::new() }
        };

        // Aggregate by chunk_id
        use std::collections::HashMap;
        let mut agg: HashMap<String, (Option<f32>, Option<f32>, Option<f32>, Option<f32>)> = HashMap::new();
        for m in tv { let e = agg.entry(m.chunk_id.0).or_default(); e.0 = Some(m.score); }
        for m in tv_and { let e = agg.entry(m.chunk_id.0).or_default(); e.1 = Some(m.score); }
        for m in tv_or { let e = agg.entry(m.chunk_id.0).or_default(); e.2 = Some(m.score); }
        for m in vec_matches { let e = agg.entry(m.chunk_id.0).or_default(); e.3 = Some(m.score); }

        // Sort by selected mode
        let mut items: Vec<(String, (Option<f32>, Option<f32>, Option<f32>, Option<f32>))> = agg.into_iter().collect();
        items.sort_by(|a, b| {
            let (tv_a, tv_and_a, tv_or_a, vec_a) = (a.1.0.unwrap_or(0.0), a.1.1.unwrap_or(0.0), a.1.2.unwrap_or(0.0), a.1.3.unwrap_or(0.0));
            let (tv_b, tv_and_b, tv_or_b, vec_b) = (b.1.0.unwrap_or(0.0), b.1.1.unwrap_or(0.0), b.1.2.unwrap_or(0.0), b.1.3.unwrap_or(0.0));
            let key_a = match self.search_mode {
                SearchMode::Hybrid => {
                    // Hybrid=fusion of TV(OR) and VEC with normalized weights
                    let denom = (self.w_text + self.w_vec).max(1e-6);
                    let wt = self.w_text / denom;
                    let wv = self.w_vec / denom;
                    wt * tv_or_a + wv * vec_a
                }
                SearchMode::Tantivy => tv_a.max(tv_and_a).max(tv_or_a),
                SearchMode::Vec => vec_a,
            };
            let key_b = match self.search_mode {
                SearchMode::Hybrid => {
                    let denom = (self.w_text + self.w_vec).max(1e-6);
                    let wt = self.w_text / denom;
                    let wv = self.w_vec / denom;
                    wt * tv_or_b + wv * vec_b
                }
                SearchMode::Tantivy => tv_b.max(tv_and_b).max(tv_or_b),
                SearchMode::Vec => vec_b,
            };
            key_b.partial_cmp(&key_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        if items.len() > top_k { items.truncate(top_k); }

        // Fetch records in batch
        let ids: Vec<chunk_model::ChunkId> = items.iter().map(|(cid, _)| chunk_model::ChunkId(cid.clone())).collect();
        let recs = match repo.get_chunks_by_ids(&ids) { Ok(r) => r, Err(e) => { self.status = format!("Fetch records failed: {e}"); return; } };
        let mut rec_map: HashMap<String, chunk_model::ChunkRecord> = HashMap::new();
        for r in recs { rec_map.insert(r.chunk_id.0.clone(), r); }

        self.results.clear();
        for (cid, (sc_tv, sc_and, sc_or, sc_vec)) in items.into_iter() {
            if let Some(rec) = rec_map.remove(&cid) {
                let file = match std::path::Path::new(&rec.source_uri).file_name().and_then(|s| s.to_str()) { Some(s) => s.to_string(), None => rec.source_uri.clone() };
                let page = match (rec.page_start, rec.page_end) {
                    (Some(s), Some(e)) if s == e => format!("#{}", s),
                    (Some(s), Some(e)) => format!("#{}-{}", s, e),
                    (Some(s), None) => format!("#{}", s),
                    _ => page_label_from_chunk_id(&rec.chunk_id.0).unwrap_or_default(),
                };
                // Flatten newlines/tabs for single-line preview, then truncate
                let flat: String = rec.text.replace(['\n', '\r', '\t'], " ");
                let mut text_preview: String = flat.chars().take(80).collect();
                if flat.chars().count() > 80 { text_preview.push('…'); }
                self.results.push(HitRow { cid: rec.chunk_id.0, file, page, text_preview, text_full: rec.text, tv: sc_tv, tv_and: sc_and, tv_or: sc_or, vec: sc_vec });
            }
        }
        self.status = format!("Results: {}", self.results.len());
    }
}

// --- helpers ---
fn derive_db_path(root: &str) -> String { format!("{}/chunks.db", root) }
fn derive_hnsw_dir(root: &str) -> String { format!("{}/hnsw", root) }
#[cfg(feature = "tantivy")]
fn derive_tantivy_dir(root: &str) -> String { format!("{}/tantivy", root) }

// Fallback: derive page label from trailing part of chunk_id like "...#12" or "...#3-4"
fn page_label_from_chunk_id(cid: &str) -> Option<String> {
    if let Some(pos) = cid.rfind('#') {
        let tail = &cid[pos + 1..];
        // parse digits or digits-digits
        let mut it = tail.splitn(2, '-');
        let a = it.next()?;
        if a.is_empty() || !a.chars().all(|c| c.is_ascii_digit()) { return None; }
        if let Some(b) = it.next() {
            if b.chars().all(|c| c.is_ascii_digit()) {
                return Some(format!("#{}-{}", a, b));
            }
            return Some(format!("#{}", a));
        }
        return Some(format!("#{}", a));
    }
    None
}

#[cfg(feature = "tantivy")]
fn rec_clone_for_tantivy(doc: &DocumentId, cid: &ChunkId, text: &str) -> ChunkRecord {
    use std::collections::BTreeMap;
    ChunkRecord {
        schema_version: SCHEMA_MAJOR,
        doc_id: DocumentId(doc.0.clone()),
        chunk_id: ChunkId(cid.0.clone()),
        source_uri: "user://input".into(),
        source_mime: "text/plain".into(),
        extracted_at: chrono::Utc::now().to_rfc3339(),
        page_start: None,
        page_end: None,
        text: text.to_string(),
        section_path: None,
        meta: BTreeMap::new(),
        extra: BTreeMap::new(),
    }
}

// --- Japanese font fallback (CJK) ------------------------------------------------------------
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

    if let Ok(custom) = env::var("EMBEDDER_DEMO_FONT") {
        paths.push(PathBuf::from(custom));
    }

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
    ] {
        paths.push(PathBuf::from(candidate));
    }

    for candidate in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ] {
        paths.push(PathBuf::from(candidate));
    }

    paths.push(PathBuf::from("fonts/NotoSansJP-Regular.otf"));
    paths
}

fn opt_fmt(v: Option<f32>) -> String {
    match v { Some(x) => format!("{:.4}", x), None => String::from("-") }
}
