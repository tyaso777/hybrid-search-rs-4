use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Instant;

use eframe::egui::{self, Button, CentralPanel, ScrollArea, Spinner, TextEdit, CollapsingHeader};
use egui_extras::{Column, TableBuilder};
use eframe::{App, CreationContext, Frame, NativeOptions};
use rfd::FileDialog;

use hybrid_service::{HybridService, ServiceConfig};
use embedding_provider::config::{default_stdio_config, ONNX_STDIO_DEFAULTS};
use chunking_store::FilterClause;
use chunking_store::{FilterKind, FilterOp, ChunkStoreRead};
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
}

#[derive(Debug)]
struct ServiceInitTask {
    rx: Receiver<Result<HybridService, String>>,
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
    svc: Option<HybridService>,
    svc_task: Option<ServiceInitTask>,

    // Insert
    input_text: String,
    doc_hint: String,
    ingest_file_path: String,

    // Search
    query: String,
    top_k: usize,
    use_hybrid: bool,
    w_text: f32,
    w_vec: f32,
    results: Vec<HitRow>,

    // UI
    tab: ActiveTab,
    status: String,
    selected_cid: Option<String>,
    selected_text: String,
    selected_display: String,
}

impl AppState {
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

            query: String::new(),
            top_k: 10,
            use_hybrid: true,
            w_text: 0.5,
            w_vec: 0.5,
            results: Vec::new(),

            tab: ActiveTab::Insert,
            status: String::new(),
            selected_cid: None,
            selected_text: String::new(),
            selected_display: String::new(),
        }
    }

    fn model_not_initialized(&self) -> bool { self.svc.is_none() && self.svc_task.is_none() }

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

        let (tx, rx) = mpsc::channel();
        self.status = "Initializing model...".into();
        self.svc_task = Some(ServiceInitTask { rx, started: Instant::now() });
        std::thread::spawn(move || {
            let res = HybridService::new(cfg).map_err(|e| e.to_string());
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
        CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, ActiveTab::Insert, "Insert");
                ui.selectable_value(&mut self.tab, ActiveTab::Search, "Search");
                ui.separator();
                if self.model_not_initialized() {
                    if ui.button("Init Model").clicked() {
                        self.start_service_init();
                    }
                } else if self.svc.is_none() {
                    ui.add(Spinner::new());
                    ui.label("Loading model...");
                } else {
                    ui.label("Model: ready");
                }
            });

            CollapsingHeader::new("Model/Store Config").default_open(true).show(ui, |ui| {
                let mut root_changed = false;
                ui.horizontal(|ui| {
                    ui.label("Store Root");
                    if ui.add(TextEdit::singleline(&mut self.store_root).desired_width(400.0)).changed() { root_changed = true; }
                    if ui.button("…").clicked() { if let Some(p) = FileDialog::new().pick_folder() { self.store_root = p.display().to_string(); root_changed = true; } }
                    if ui.button("Reset").clicked() { self.store_root = "target/demo/store".into(); root_changed = true; }
                });
                if root_changed { self.refresh_store_paths(); }
                ui.horizontal(|ui| { ui.label("DB"); ui.label(&self.db_path); });
                ui.horizontal(|ui| { ui.label("HNSW"); ui.label(&self.hnsw_dir); });
                #[cfg(feature = "tantivy")]
                ui.horizontal(|ui| { ui.label("Tantivy"); ui.label(&self.tantivy_dir); });
                ui.separator();
                ui.collapsing("Danger zone", |ui| {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Deletes DB and indexes (HNSW/Tantivy)").color(egui::Color32::LIGHT_RED));
                        if ui.button(egui::RichText::new("Delete DB & Indexes").color(egui::Color32::RED)).clicked() {
                            self.delete_store_files();
                        }
                    });
                });
                ui.separator();
                ui.horizontal(|ui| { ui.label("Model"); ui.add(TextEdit::singleline(&mut self.model_path).desired_width(400.0)); if ui.button("…").clicked() { if let Some(p) = FileDialog::new().add_filter("ONNX", &["onnx"]).pick_file() { self.model_path = p.display().to_string(); } } });
                ui.horizontal(|ui| { ui.label("Tokenizer"); ui.add(TextEdit::singleline(&mut self.tokenizer_path).desired_width(400.0)); if ui.button("…").clicked() { if let Some(p) = FileDialog::new().add_filter("JSON", &["json"]).pick_file() { self.tokenizer_path = p.display().to_string(); } } });
                ui.horizontal(|ui| { ui.label("Runtime DLL"); ui.add(TextEdit::singleline(&mut self.runtime_path).desired_width(400.0)); if ui.button("…").clicked() { if let Some(p) = FileDialog::new().pick_file() { self.runtime_path = p.display().to_string(); } } });
                ui.horizontal(|ui| { ui.label("Dim"); ui.add(TextEdit::singleline(&mut self.embedding_dimension).desired_width(80.0)); ui.label("MaxTokens"); ui.add(TextEdit::singleline(&mut self.max_tokens).desired_width(80.0)); });
                ui.horizontal(|ui| { ui.checkbox(&mut self.preload_model_to_memory, "Preload model into memory"); });
            });

            ui.separator();
            match self.tab {
                ActiveTab::Insert => self.ui_insert(ui),
                ActiveTab::Search => self.ui_search(ui),
            }

            ui.separator();
            if !self.status.is_empty() { ui.label(&self.status); }
        });
    }
}

impl AppState {
    fn ui_insert(&mut self, ui: &mut egui::Ui) {
        ui.heading("Insert");
        ui.horizontal(|ui| {
            ui.label("Doc Hint");
            ui.add(TextEdit::singleline(&mut self.doc_hint).desired_width(200.0));
        });
        ui.horizontal(|ui| {
            ui.label("Text");
            ui.add(TextEdit::multiline(&mut self.input_text).desired_rows(4).desired_width(600.0));
        });
        ui.horizontal(|ui| {
            if ui.add(Button::new("Insert Text")).clicked() { self.do_insert_text(); }
            ui.separator();
            ui.add(TextEdit::singleline(&mut self.ingest_file_path).desired_width(400.0));
            if ui.button("Choose File").clicked() { if let Some(p) = FileDialog::new().pick_file() { self.ingest_file_path = p.display().to_string(); } }
            if ui.add(Button::new("Ingest File")).clicked() { self.do_ingest_file(); }
        });
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
        match svc.ingest_file(&path_owned, if self.doc_hint.trim().is_empty() { None } else { Some(self.doc_hint.trim()) }) {
            Ok(()) => {
                // Best-effort Tantivy upsert for ingested file
                #[cfg(feature = "tantivy")]
                {
                    if let Err(err) = self.ensure_tantivy_open() { self.status = format!("Tantivy init failed: {err}"); return; }
                    use chunking_store::sqlite_repo::SqliteRepo;
                    let repo = match SqliteRepo::open(self.db_path.trim()) { Ok(r) => r, Err(e) => { self.status = format!("Open DB failed: {e}"); return; } };
                    let doc = if self.doc_hint.trim().is_empty() { path_owned.clone() } else { self.doc_hint.trim().to_string() };
                    let filter = FilterClause { kind: FilterKind::Must, op: FilterOp::DocIdEq(doc) };
                    let ids = match repo.list_chunk_ids_by_filter(&[filter], 10_000, 0) { Ok(v) => v, Err(_) => Vec::new() };
                    if !ids.is_empty() {
                        if let Ok(recs) = repo.get_chunks_by_ids(&ids) {
                            if let Some(idx) = &self.tantivy {
                                let _ = idx.upsert_records(&recs);
                            }
                        }
                    }
                }
                self.status = format!("Ingested file: {}", path_owned);
            }
            Err(e) => { self.status = format!("Ingest failed: {e}"); }
        }
    }

    fn ui_search(&mut self, ui: &mut egui::Ui) {
        ui.push_id("search_panel", |ui| {
            ui.heading("Search");
            // Row 1: Query + Search
            ui.horizontal(|ui| {
                ui.label("Query");
                ui.add(TextEdit::singleline(&mut self.query).desired_width(400.0).id_source("search_query"));
                if ui.add(Button::new("Search")).clicked() { self.do_search_now(); }
            });
            // Row 2: Options (TopK / Hybrid / weights)
            ui.horizontal(|ui| {
                ui.label("TopK");
                let mut topk_str = self.top_k.to_string();
                if ui.add(TextEdit::singleline(&mut topk_str).desired_width(60.0).id_source("search_topk")).changed() {
                    self.top_k = topk_str.parse().unwrap_or(10);
                }
                ui.checkbox(&mut self.use_hybrid, "Hybrid");
                if self.use_hybrid {
                    ui.label("w_text"); ui.add(egui::DragValue::new(&mut self.w_text).speed(0.1).clamp_range(0.0..=1.0));
                    ui.label("w_vec"); ui.add(egui::DragValue::new(&mut self.w_vec).speed(0.1).clamp_range(0.0..=1.0));
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
        let _ = repo.maybe_rebuild_fts();
        let opts = SearchOptions { top_k, fetch_factor: 10 };

        // Tantivy queries
        #[cfg(feature = "tantivy")]
        let (tv, tv_and, tv_or) = {
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
        let vec_matches: Vec<chunking_store::TextMatch> = if self.use_hybrid {
            if let Some(svc) = &self.svc {
                match svc.search_hybrid(q, top_k, filters, 0.0, 1.0) {
                    Ok(hits) => hits.into_iter().map(|h| chunking_store::TextMatch { chunk_id: chunk_model::ChunkId(h.chunk.chunk_id.0), score: h.score, raw_score: h.score }).collect(),
                    Err(_) => Vec::new(),
                }
            } else { Vec::new() }
        } else { Vec::new() };

        // Aggregate by chunk_id
        use std::collections::HashMap;
        let mut agg: HashMap<String, (Option<f32>, Option<f32>, Option<f32>, Option<f32>)> = HashMap::new();
        for m in tv { let e = agg.entry(m.chunk_id.0).or_default(); e.0 = Some(m.score); }
        for m in tv_and { let e = agg.entry(m.chunk_id.0).or_default(); e.1 = Some(m.score); }
        for m in tv_or { let e = agg.entry(m.chunk_id.0).or_default(); e.2 = Some(m.score); }
        for m in vec_matches { let e = agg.entry(m.chunk_id.0).or_default(); e.3 = Some(m.score); }

        // Sort keys by max score desc and truncate to top_k
        let mut items: Vec<(String, (Option<f32>, Option<f32>, Option<f32>, Option<f32>))> = agg.into_iter().collect();
        items.sort_by(|a, b| {
            let max_a = a.1.0.unwrap_or(0.0).max(a.1.1.unwrap_or(0.0)).max(a.1.2.unwrap_or(0.0)).max(a.1.3.unwrap_or(0.0));
            let max_b = b.1.0.unwrap_or(0.0).max(b.1.1.unwrap_or(0.0)).max(b.1.2.unwrap_or(0.0)).max(b.1.3.unwrap_or(0.0));
            max_b.partial_cmp(&max_a).unwrap_or(std::cmp::Ordering::Equal)
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
                let text_preview: String = rec.text.chars().take(80).collect();
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
