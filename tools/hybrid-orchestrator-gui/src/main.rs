use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Instant;

use chrono::Utc;
use chunk_model::{ChunkId, ChunkRecord, DocumentId, SCHEMA_MAJOR};
use chunking_store::fts5_index::Fts5Index;
#[cfg(feature = "tantivy")]
use chunking_store::tantivy_index::TantivyIndex;
use chunking_store::hnsw_index::HnswIndex;
use chunking_store::orchestrator::ingest_chunks_orchestrated;
use chunking_store::sqlite_repo::SqliteRepo;
use chunking_store::{SearchOptions, VectorSearcher, ChunkStoreRead};
use eframe::egui::{self, Button, CentralPanel, ScrollArea, Spinner, TextEdit};
use egui_extras::{Column, TableBuilder};
use eframe::{App, CreationContext, Frame, NativeOptions};
use embedding_provider::config::{default_stdio_config, ONNX_STDIO_DEFAULTS};
use embedding_provider::embedder::{Embedder, OnnxStdIoConfig, OnnxStdIoEmbedder};
use rfd::FileDialog;
use calamine::{open_workbook_auto, Reader};

fn main() -> eframe::Result<()> {
    let options = NativeOptions::default();
    eframe::run_native(
        "Hybrid Orchestrator",
        options,
        Box::new(|cc| Box::new(AppState::new(cc))),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveTab {
    Insert,
    Search,
    ExcelIngest,
}

#[derive(Debug)]
struct ModelInitTask {
    rx: Receiver<Result<OnnxStdIoEmbedder, String>>,
    started: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingAction {
    Insert,
    HybridSearch,
}

struct AppState {
    // Model config
    model_path: String,
    tokenizer_path: String,
    runtime_path: String,
    embedding_dimension: String,
    max_tokens: String,
    embedder: Option<OnnxStdIoEmbedder>,
    model_task: Option<ModelInitTask>,
    pending_action: Option<PendingAction>,
    #[cfg(feature = "tantivy")]
    tantivy: Option<TantivyIndex>,

    // Store/index config
    db_path: String,
    hnsw_dir: String,
    #[cfg(feature = "tantivy")]
    tantivy_dir: String,

    // Insert
    input_text: String,
    doc_hint: String,

    // Search
    query: String,
    top_k: usize,
    use_hybrid: bool,
    results: Vec<HitRow>,

    // Excel ingest
    input_excel_path: String,
    excel_skip_header: bool,
    excel_batch_size: usize,
    // Danger zone per resource
    reset_db_input: String,
    reset_hnsw_input: String,
    #[cfg(feature = "tantivy")]
    reset_tantivy_input: String,

    // UI
    tab: ActiveTab,
    status: String,
    // Selection for details view
    selected_cid: Option<String>,
    selected_text: String,
}

impl AppState {
    fn new(cc: &CreationContext<'_>) -> Self {
        install_japanese_fallback_fonts(&cc.egui_ctx);
        let defaults = default_stdio_config();
        let db_default = String::from("target/demo/chunks.db");
        Self {
            model_path: defaults.model_path.display().to_string(),
            tokenizer_path: defaults.tokenizer_path.display().to_string(),
            runtime_path: defaults.runtime_library_path.display().to_string(),
            embedding_dimension: ONNX_STDIO_DEFAULTS.embedding_dimension.to_string(),
            max_tokens: ONNX_STDIO_DEFAULTS.max_input_tokens.to_string(),
            embedder: None,
            model_task: None,
            pending_action: None,
            #[cfg(feature = "tantivy")]
            tantivy: None,

            db_path: db_default.clone(),
            hnsw_dir: derive_hnsw_dir(&db_default),
            #[cfg(feature = "tantivy")]
            tantivy_dir: derive_tantivy_dir(&db_default),

            input_text: String::new(),
            doc_hint: String::new(),

            query: String::new(),
            top_k: 10,
            use_hybrid: true,
            results: Vec::new(),

            input_excel_path: String::new(),
            excel_skip_header: true,
            excel_batch_size: 128,
            reset_db_input: String::new(),
            reset_hnsw_input: String::new(),
            #[cfg(feature = "tantivy")]
            reset_tantivy_input: String::new(),

            tab: ActiveTab::Insert,
            status: "Ready".into(),
            selected_cid: None,
            selected_text: String::new(),
        }
    }

    fn build_config(&self) -> Result<OnnxStdIoConfig, String> {
        let dim: usize = self
            .embedding_dimension
            .trim()
            .parse()
            .map_err(|e| format!("Invalid dimension: {e}"))?;
        let max_len: usize = self
            .max_tokens
            .trim()
            .parse()
            .map_err(|e| format!("Invalid max tokens: {e}"))?;
        Ok(OnnxStdIoConfig {
            model_path: PathBuf::from(self.model_path.trim()),
            tokenizer_path: PathBuf::from(self.tokenizer_path.trim()),
            runtime_library_path: PathBuf::from(self.runtime_path.trim()),
            dimension: dim,
            max_input_length: max_len,
            embedding_model_id: ONNX_STDIO_DEFAULTS.embedding_model_id.into(),
            text_repr_version: ONNX_STDIO_DEFAULTS.text_repr_version.into(),
        })
    }

    fn start_model_init(&mut self, context_hint: &str) {
        if self.model_task.is_some() { return; }
        let cfg = match self.build_config() { Ok(c) => c, Err(err) => { self.status = err; return; } };
        let (tx, rx) = mpsc::channel();
        self.status = if context_hint.is_empty() { "Initializing model...".into() } else { format!("Initializing model ({})...", context_hint) };
        std::thread::spawn(move || {
            let res = OnnxStdIoEmbedder::new(cfg).map_err(|e| format!("{e}"));
            let _ = tx.send(res);
        });
        self.model_task = Some(ModelInitTask { rx, started: Instant::now() });
    }

    fn ensure_embedder_or_queue(&mut self, action: PendingAction) -> bool {
        if self.embedder.is_some() { return true; }
        if self.model_task.is_some() {
            self.pending_action = Some(action);
            self.status = "Waiting for model init...".into();
            return false;
        }
        self.pending_action = Some(action);
        self.start_model_init("");
        false
    }

    fn do_insert_now(&mut self) {
        let text = self.input_text.trim().to_string();
        if text.is_empty() { self.status = "Enter some text to insert".into(); return; }
        let db = self.db_path.trim();
        if db.is_empty() { self.status = "Enter DB path".into(); return; }
        let hdir = if self.hnsw_dir.trim().is_empty() { derive_hnsw_dir(db) } else { self.hnsw_dir.trim().to_string() };
        let embedder = match self.embedder.as_ref() { Some(e) => e, None => { self.status = "Model not initialized".into(); return; } };

        // Embed
        let vector = match embedder.embed(&text) { Ok(v) => v, Err(e) => { self.status = format!("Embedding failed: {e}"); return; } };

        // Open store
        if let Some(parent) = PathBuf::from(db).parent() { let _ = fs::create_dir_all(parent); }
        let mut repo = match SqliteRepo::open(db) { Ok(r) => r, Err(e) => { self.status = format!("Open DB failed: {e}"); return; } };
        let _ = repo.maybe_rebuild_fts();

        // Build record
        let (doc_id, chunk_id) = make_ids_from_text(self.doc_hint.trim(), &text);
        let mut meta = std::collections::BTreeMap::new();
        meta.insert("ingest.ts".into(), Utc::now().to_rfc3339());
        meta.insert("len".into(), text.len().to_string());
        let rec = ChunkRecord {
            schema_version: SCHEMA_MAJOR,
            doc_id: doc_id.clone(),
            chunk_id: chunk_id.clone(),
            source_uri: "user://input".into(),
            source_mime: "text/plain".into(),
            extracted_at: Utc::now().to_rfc3339(),
            text: text.clone(),
            section_path: vec![],
            meta,
            extra: std::collections::BTreeMap::new(),
        };

        // Indexes
        let fts = Fts5Index::new();
        let text_indexes: [&dyn chunking_store::TextIndexMaintainer; 1] = [&fts];
        let mut hnsw = if PathBuf::from(&hdir).join("map.tsv").exists() { match HnswIndex::load(&hdir, vector.len()) { Ok(h) => h, Err(e) => { self.status = format!("Load HNSW failed: {e}"); return; } } } else { HnswIndex::new(vector.len(), 10_000) };
        let mut vec_indexes: [&mut dyn chunking_store::VectorIndexMaintainer; 1] = [&mut hnsw];
        let vectors = vec![(chunk_id.clone(), vector)];
        if let Err(e) = ingest_chunks_orchestrated(&mut repo, &[rec], &text_indexes, &mut vec_indexes, Some(&vectors)) { self.status = format!("Ingest failed: {e}"); return; }
        if let Err(e) = hnsw.save(&hdir) { self.status = format!("Save HNSW failed: {e}"); return; }

        // Upsert into Tantivy when enabled (in-memory index)
        #[cfg(feature = "tantivy")]
        {
            let tdir = if self.tantivy_dir.trim().is_empty() { derive_tantivy_dir(db) } else { self.tantivy_dir.trim().to_string() };
            // Lazily open/create Tantivy index on disk
            if self.tantivy.is_none() {
                match TantivyIndex::open_or_create_dir(&tdir) {
                    Ok(idx) => self.tantivy = Some(idx),
                    Err(err) => { self.status = format!("Init Tantivy failed: {err}"); return; }
                }
            }
            if let Some(idx) = &self.tantivy {
                if let Err(e) = idx.upsert_records(&[rec_clone_for_tantivy(&doc_id, &chunk_id, &text)]) {
                    self.status = format!("Tantivy upsert failed: {e}");
                    return;
                }
            }
        }
        self.status = format!("Inserted chunk {} (doc={})", chunk_id.0, doc_id.0);
    }

    fn do_search_now(&mut self) {
        let db = self.db_path.trim();
        if db.is_empty() { self.status = "Enter DB path".into(); return; }
        let q = self.query.trim();
        if q.is_empty() { self.status = "Enter query".into(); return; }
        let repo = match SqliteRepo::open(db) { Ok(r) => r, Err(e) => { self.status = format!("Open DB failed: {e}"); return; } };
        let _ = repo.maybe_rebuild_fts();
        let fts = Fts5Index::new();
        let opts = SearchOptions { top_k: self.top_k, fetch_factor: 10 };

        // Run all available engines; combine and display separate scores.
        // Always run FTS5. Run vector if HNSW snapshot exists. Run Tantivy if available and initialized.
        let mut fts_matches = chunking_store::TextSearcher::search_ids(&fts, &repo, q, &[], &opts);

        // Vector
        let mut vec_matches = if self.use_hybrid {
            let hdir = if self.hnsw_dir.trim().is_empty() { derive_hnsw_dir(db) } else { self.hnsw_dir.trim().to_string() };
            if let Some(e) = &self.embedder {
                if let Ok(qvec) = e.embed(q) {
                    if PathBuf::from(&hdir).join("map.tsv").exists() {
                        if let Ok(h) = HnswIndex::load(&hdir, qvec.len()) {
                            VectorSearcher::knn_ids(&h, &repo, &qvec, &[], &opts)
                        } else { Vec::new() }
                    } else { Vec::new() }
                } else { Vec::new() }
            } else { Vec::new() }
        } else { Vec::new() };

        // Tantivy
        #[cfg(feature = "tantivy")]
        let mut tv_matches = {
            if self.tantivy.is_none() {
                let tdir = if self.tantivy_dir.trim().is_empty() { derive_tantivy_dir(db) } else { self.tantivy_dir.trim().to_string() };
                if let Ok(tv) = TantivyIndex::open_or_create_dir(&tdir) {
                    // Best-effort populate once (first open) from DB
                    let batch = 1000;
                    let mut offset = 0usize;
                    let empty_filters: &[chunking_store::FilterClause] = &[];
                    loop {
                        let ids = match repo.list_chunk_ids_by_filter(empty_filters, batch, offset) { Ok(v) => v, Err(_) => Vec::new() };
                        if ids.is_empty() { break; }
                        offset += ids.len();
                        if let Ok(recs) = repo.get_chunks_by_ids(&ids) { let _ = tv.upsert_records(&recs); }
                    }
                    self.tantivy = Some(tv);
                }
            }
            if let Some(tv) = &self.tantivy { chunking_store::TextSearcher::search_ids(tv, &repo, q, &[], &opts) } else { Vec::new() }
        };
        #[cfg(not(feature = "tantivy"))]
        let mut tv_matches: Vec<chunking_store::TextMatch> = Vec::new();

        // Combine (ranking): keep FTS + Vector fused score as before; show all scores separately.
        let w_text = 0.5f32;
        let w_vec = 0.5f32;
        use std::collections::HashMap;
        let mut rows: HashMap<String, HitRow> = HashMap::new();
        for m in fts_matches.drain(..) {
            let entry = rows.entry(m.chunk_id.0).or_insert_with(HitRow::empty);
            entry.fts = Some(m.score);
            entry.combined += w_text * m.score;
        }
        for m in vec_matches.drain(..) {
            let entry = rows.entry(m.chunk_id.0).or_insert_with(HitRow::empty);
            entry.vec = Some(m.score);
            entry.combined += w_vec * m.score;
        }
        for m in tv_matches.drain(..) {
            let entry = rows.entry(m.chunk_id.0).or_insert_with(HitRow::empty);
            entry.tv = Some(m.score);
            // Not included in combined ordering (can change if desired)
        }

        let mut pairs: Vec<(String, HitRow)> = rows.into_iter().collect();
        pairs.sort_by(|a, b| b.1.combined.partial_cmp(&a.1.combined).unwrap_or(std::cmp::Ordering::Equal));
        if pairs.len() > self.top_k { pairs.truncate(self.top_k); }
        let ids: Vec<ChunkId> = pairs.iter().map(|(cid, _)| ChunkId(cid.clone())).collect();
        let recs = match repo.get_chunks_by_ids(&ids) { Ok(r) => r, Err(e) => { self.status = format!("Fetch chunks failed: {e}"); return; } };

        let map: HashMap<String, HitRow> = pairs.into_iter().collect();
        self.results = recs.into_iter().map(|rec| {
            let mut row = map.get(&rec.chunk_id.0).cloned().unwrap_or_else(HitRow::empty);
            row.cid = rec.chunk_id.0;
            let full_text = rec.text;
            row.preview = truncate_chars(&full_text, 80);
            row.full = full_text;
            row
        }).collect();
        self.status = format!("Hits: {}", self.results.len());
    }

    fn run_excel_ingest(&mut self) {
        let db = self.db_path.trim();
        if db.is_empty() { self.status = "Enter DB path".into(); return; }
        let path = self.input_excel_path.trim();
        if path.is_empty() { self.status = "Pick Excel workbook".into(); return; }
        let embedder = match self.embedder.as_ref() { Some(e) => e, None => { self.status = "Model not initialized".into(); return; } };

        // Open DB and indexes
        if let Some(parent) = PathBuf::from(db).parent() { let _ = std::fs::create_dir_all(parent); }
        let mut repo = match SqliteRepo::open(db) { Ok(r) => r, Err(e) => { self.status = format!("Open DB failed: {e}"); return; } };
        let _ = repo.maybe_rebuild_fts();
        let fts = Fts5Index::new();

        // HNSW
        let hdir = if self.hnsw_dir.trim().is_empty() { derive_hnsw_dir(db) } else { self.hnsw_dir.trim().to_string() };
        let mut hnsw = if PathBuf::from(&hdir).join("map.tsv").exists() { match HnswIndex::load(&hdir, embedder.info().dimension) { Ok(h) => h, Err(_) => HnswIndex::new(embedder.info().dimension, 10_000) } } else { HnswIndex::new(embedder.info().dimension, 10_000) };

        // Tantivy
        #[cfg(feature = "tantivy")]
        if self.tantivy.is_none() {
            let tdir = if self.tantivy_dir.trim().is_empty() { derive_tantivy_dir(db) } else { self.tantivy_dir.trim().to_string() };
            if let Ok(tv) = TantivyIndex::open_or_create_dir(&tdir) { self.tantivy = Some(tv); }
        }

        // Read Excel
        let mut workbook = match open_workbook_auto(path) { Ok(wb) => wb, Err(e) => { self.status = format!("Open Excel failed: {e}"); return; } };
        let range = match workbook.worksheet_range_at(0) { Some(Ok(r)) => r, _ => { self.status = "Failed to read first worksheet".into(); return; } };
        let mut rows: Vec<String> = Vec::new();
        for (i, row) in range.rows().enumerate() {
            if i == 0 && self.excel_skip_header { continue; }
            let text = row.get(0).map(|cell| cell.to_string()).unwrap_or_default().trim().to_string();
            if !text.is_empty() { rows.push(text); }
        }
        if rows.is_empty() { self.status = "No rows to ingest".into(); return; }

        let batch = self.excel_batch_size.max(1);
        let mut done = 0usize;
        while done < rows.len() {
            let end = usize::min(done + batch, rows.len());
            let slice = &rows[done..end];
            // Embed batch
            let refs: Vec<&str> = slice.iter().map(String::as_str).collect();
            let vectors = match embedder.embed_batch(&refs) { Ok(v) => v, Err(e) => { self.status = format!("Embedding failed: {e}"); return; } };
            // Build records and vector pairs
            let mut records: Vec<ChunkRecord> = Vec::with_capacity(slice.len());
            let mut pairs: Vec<(ChunkId, Vec<f32>)> = Vec::with_capacity(slice.len());
            for (text, vec) in slice.iter().zip(vectors.into_iter()) {
                let (doc_id, chunk_id) = make_ids_from_text("", text);
                let mut meta = std::collections::BTreeMap::new();
                meta.insert("ingest.ts".into(), chrono::Utc::now().to_rfc3339());
                meta.insert("len".into(), text.len().to_string());
                let rec = ChunkRecord {
                    schema_version: SCHEMA_MAJOR,
                    doc_id: doc_id.clone(),
                    chunk_id: chunk_id.clone(),
                    source_uri: "excel://row".into(),
                    source_mime: "text/plain".into(),
                    extracted_at: chrono::Utc::now().to_rfc3339(),
                    text: text.clone(),
                    section_path: Vec::new(),
                    meta,
                    extra: std::collections::BTreeMap::new(),
                };
                records.push(rec);
                pairs.push((chunk_id, vec));
            }

            // Orchestrated ingest into DB + FTS + HNSW
            let mut vec_indexes: [&mut dyn chunking_store::VectorIndexMaintainer; 1] = [&mut hnsw];
            let text_indexes: [&dyn chunking_store::TextIndexMaintainer; 1] = [&fts];
            if let Err(e) = ingest_chunks_orchestrated(&mut repo, &records, &text_indexes, &mut vec_indexes, Some(&pairs)) { self.status = format!("Ingest failed: {e}"); return; }
            #[cfg(feature = "tantivy")]
            if let Some(tv) = &self.tantivy { let _ = tv.upsert_records(&records); }

            done = end;
            self.status = format!("Ingested {}/{} rows", done, rows.len());
        }
        let _ = hnsw.save(&hdir);
        self.status = format!("Excel ingest complete: {} rows", rows.len());
    }

    fn reset_db_only(&mut self) {
        let db = self.db_path.trim();
        if db.is_empty() { self.status = "Enter DB path first".into(); return; }
        let db_path = PathBuf::from(db);
        if db_path.exists() {
            if let Err(e) = std::fs::remove_file(&db_path) { self.status = format!("Failed to remove DB: {e}"); return; }
            let wal = db_path.with_extension("db-wal");
            let shm = db_path.with_extension("db-shm");
            let _ = std::fs::remove_file(wal);
            let _ = std::fs::remove_file(shm);
        }
        self.status = "SQLite DB removed".into();
    }

    fn reset_hnsw_only(&mut self) {
        let db = self.db_path.trim();
        let hdir = if self.hnsw_dir.trim().is_empty() { derive_hnsw_dir(db) } else { self.hnsw_dir.trim().to_string() };
        if !hdir.trim().is_empty() {
            let p = PathBuf::from(&hdir);
            if p.exists() { let _ = std::fs::remove_dir_all(&p); }
        }
        self.status = "HNSW dir removed".into();
    }

    #[cfg(feature = "tantivy")]
    fn reset_tantivy_only(&mut self) {
        let db = self.db_path.trim();
        let tdir = if self.tantivy_dir.trim().is_empty() { derive_tantivy_dir(db) } else { self.tantivy_dir.trim().to_string() };
        if !tdir.trim().is_empty() {
            let p = PathBuf::from(&tdir);
            if p.exists() { let _ = std::fs::remove_dir_all(&p); }
        }
        self.tantivy = None;
        self.status = "Tantivy dir removed".into();
    }

    fn reset_all(&mut self) {
        // Delete DB
        self.reset_db_only();
        // Delete HNSW
        self.reset_hnsw_only();
        // Delete Tantivy
        #[cfg(feature = "tantivy")]
        { self.reset_tantivy_only(); }
        self.results.clear();
        self.selected_cid = None;
        self.selected_text.clear();
        self.status = "All data removed (DB/HNSW/Tantivy)".into();
    }
}

impl App for AppState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        CentralPanel::default().show(ctx, |ui| {
            ScrollArea::vertical().id_source("root_scroll").auto_shrink([false; 2]).show(ui, |ui| {
                ui.heading("Model Configuration");
                ui.horizontal(|ui| {
                    ui.label("Model ONNX:");
                    ui.add(TextEdit::singleline(&mut self.model_path).desired_width(400.0));
                    if ui.add(Button::new("Browse")).clicked() {
                        if let Some(path) = FileDialog::new().add_filter("ONNX", &["onnx"]).pick_file() {
                            self.model_path = path.display().to_string();
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Tokenizer JSON:");
                    ui.add(TextEdit::singleline(&mut self.tokenizer_path).desired_width(400.0));
                    if ui.add(Button::new("Browse")).clicked() {
                        if let Some(path) = FileDialog::new().add_filter("JSON", &["json"]).pick_file() {
                            self.tokenizer_path = path.display().to_string();
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("ONNX Runtime DLL:");
                    ui.add(TextEdit::singleline(&mut self.runtime_path).desired_width(400.0));
                    if ui.add(Button::new("Browse")).clicked() {
                        if let Some(path) = FileDialog::new().add_filter("DLL", &["dll"]).pick_file() {
                            self.runtime_path = path.display().to_string();
                        }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Dimension:");
                    ui.add(TextEdit::singleline(&mut self.embedding_dimension).desired_width(120.0));
                    ui.label("Max tokens:");
                    ui.add(TextEdit::singleline(&mut self.max_tokens).desired_width(120.0));
                    let init_btn = ui.add_enabled(self.model_task.is_none(), Button::new("Initialize Model"));
                    if init_btn.clicked() { self.start_model_init(""); }
                    if self.model_task.is_some() { ui.add(Spinner::new()); ui.label("Initializing model..."); }
                });

                ui.separator();
                // Poll model init
                if let Some(task) = &self.model_task {
                    match task.rx.try_recv() {
                        Ok(Ok(embedder)) => {
                            let elapsed = task.started.elapsed().as_secs_f32();
                            self.embedder = Some(embedder);
                            self.status = format!("Model initialized in {:.2}s", elapsed);
                            self.model_task = None;
                            if let Some(a) = self.pending_action.take() {
                                match a { PendingAction::Insert => self.do_insert_now(), PendingAction::HybridSearch => self.do_search_now(), }
                            }
                        }
                        Ok(Err(err)) => { self.status = format!("Init failed: {err}"); self.model_task = None; }
                        Err(TryRecvError::Empty) => { ctx.request_repaint(); }
                        Err(TryRecvError::Disconnected) => { self.status = "Init failed (disconnected)".into(); self.model_task = None; }
                    }
                }

                ui.label(format!("Status: {}", self.status));
                ui.separator();

                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.tab, ActiveTab::Insert, "Insert");
                    ui.selectable_value(&mut self.tab, ActiveTab::ExcelIngest, "Excel Ingest");
                    ui.selectable_value(&mut self.tab, ActiveTab::Search, "Search");
                });

                ui.separator();

                ui.heading("Store / Index Configuration");
                ui.horizontal(|ui| {
                    ui.label("SQLite DB:");
                    ui.add(TextEdit::singleline(&mut self.db_path).desired_width(420.0));
                    if ui.add(Button::new("Browse")).clicked() {
                        if let Some(path) = FileDialog::new().add_filter("DB", &["db"]).save_file() {
                            self.db_path = path.display().to_string();
                            self.hnsw_dir = derive_hnsw_dir(&self.db_path);
                            #[cfg(feature = "tantivy")]
                            { self.tantivy_dir = derive_tantivy_dir(&self.db_path); }
                        }
                    }
                });
                // 単独のDB削除は廃止（全削除ボタンは下のTantivyセクションに集約）

                ui.horizontal(|ui| {
                    ui.label("HNSW Dir:");
                    ui.add(TextEdit::singleline(&mut self.hnsw_dir).desired_width(420.0));
                    if ui.add(Button::new("Pick Dir")).clicked() {
                        if let Some(path) = FileDialog::new().pick_folder() { self.hnsw_dir = path.display().to_string(); }
                    }
                });
                // 単独のHNSW削除は廃止（全削除ボタンは下のTantivyセクションに集約）
                #[cfg(feature = "tantivy")]
                ui.horizontal(|ui| {
                    ui.label("Tantivy Dir:");
                    ui.add(TextEdit::singleline(&mut self.tantivy_dir).desired_width(420.0));
                    if ui.add(Button::new("Pick Dir")).clicked() {
                        if let Some(path) = FileDialog::new().pick_folder() { self.tantivy_dir = path.display().to_string(); }
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Irreversible: type 'RESET' to enable Delete All (DB/HNSW/Tantivy)");
                    #[allow(unused_mut)]
                    let mut _tmp = String::new();
                    #[cfg(feature = "tantivy")]
                    let txt_ref = &mut self.reset_tantivy_input;
                    #[cfg(not(feature = "tantivy"))]
                    let txt_ref = &mut tmp;
                    ui.add(TextEdit::singleline(txt_ref).desired_width(120.0));
                    #[cfg(feature = "tantivy")]
                    {
                        let can = self.reset_tantivy_input.trim() == "RESET";
                        let btn = ui.add_enabled(can, Button::new("Delete All (DB/HNSW/Tantivy)"));
                        if btn.clicked() { self.reset_all(); self.reset_tantivy_input.clear(); }
                    }
                });

                ui.separator();

                match self.tab {
                    ActiveTab::Insert => {
                        ui.heading("Insert Text");
                        ui.add(TextEdit::multiline(&mut self.input_text).desired_rows(4).desired_width(620.0));
                        ui.horizontal(|ui| {
                            ui.label("Doc ID (optional):");
                            ui.add(TextEdit::singleline(&mut self.doc_hint).desired_width(220.0));
                            let insert_btn = ui.button("Insert");
                            if insert_btn.clicked() {
                                if self.ensure_embedder_or_queue(PendingAction::Insert) { self.do_insert_now(); }
                            }
                        });
                    }
                    ActiveTab::Search => {
                        ui.heading("Search");
                        ui.add(TextEdit::singleline(&mut self.query).desired_width(420.0));
                        ui.horizontal(|ui| {
                            ui.label("Top K:");
                            ui.add(egui::Slider::new(&mut self.top_k, 1..=100).text("K"));
                            ui.checkbox(&mut self.use_hybrid, "Embed query vector (for HNSW)");
                            let search_btn = ui.button("Search");
                            if search_btn.clicked() {
                                if self.use_hybrid && self.embedder.is_none() {
                                    if self.ensure_embedder_or_queue(PendingAction::HybridSearch) { self.do_search_now(); }
                                } else {
                                    self.do_search_now();
                                }
                            }
                        });
                        ui.separator();
                        ui.heading("Results");
                        let header_height = 24.0;
                        let row_height = 24.0;
                        ScrollArea::vertical().id_source("results_scroll").max_height(320.0).show(ui, |ui| {
                            TableBuilder::new(ui)
                                .striped(true)
                                .column(Column::exact(36.0))
                                .column(Column::exact(220.0))
                                .column(Column::exact(70.0))
                                .column(Column::exact(70.0))
                                .column(Column::exact(70.0))
                                .column(Column::exact(80.0))
                                .column(Column::remainder())
                                .header(header_height, |mut header| {
                                    header.col(|ui| { ui.label("#"); });
                                    header.col(|ui| { ui.label("Chunk ID"); });
                                    header.col(|ui| { ui.label("FTS"); });
                                    header.col(|ui| { ui.label("TV"); });
                                    header.col(|ui| { ui.label("VEC"); });
                                    header.col(|ui| { ui.label("Comb"); });
                                    header.col(|ui| { ui.label("Preview"); });
                                })
                                .body(|mut body| {
                                    for (i, row) in self.results.iter().enumerate() {
                                    body.row(row_height, |mut trow| {
                                        let selected = self.selected_cid.as_deref() == Some(row.cid.as_str());
                                        let mut row_rect: Option<egui::Rect> = None;

                                        trow.col(|ui| {
                                            let mut rt = egui::RichText::new(format!("{:>2}", i+1)).monospace();
                                            if selected { rt = rt.strong(); }
                                            let mut lbl = egui::Label::new(rt).sense(egui::Sense::click());
                                            lbl = lbl.wrap(false).truncate(true);
                                            let r = ui.add(lbl);
                                            if r.clicked() { self.selected_cid = Some(row.cid.clone()); self.selected_text = row.full.clone(); }
                                            row_rect = Some(row_rect.map_or(r.rect, |rr| rr.union(r.rect)));
                                        });
                                        trow.col(|ui| {
                                            let mut rt = egui::RichText::new(row.cid.clone()).monospace();
                                            if selected { rt = rt.strong(); }
                                            let mut lbl = egui::Label::new(rt).sense(egui::Sense::click());
                                            lbl = lbl.wrap(false).truncate(true);
                                            let r = ui.add(lbl);
                                            if r.clicked() { self.selected_cid = Some(row.cid.clone()); self.selected_text = row.full.clone(); }
                                            row_rect = Some(row_rect.map_or(r.rect, |rr| rr.union(r.rect)));
                                        });
                                        trow.col(|ui| {
                                            let mut rt = egui::RichText::new(format!("{:.4}", row.fts.unwrap_or(0.0))).monospace();
                                            if selected { rt = rt.strong(); }
                                            let mut lbl = egui::Label::new(rt).sense(egui::Sense::click());
                                            lbl = lbl.wrap(false).truncate(true);
                                            let r = ui.add(lbl);
                                            if r.clicked() { self.selected_cid = Some(row.cid.clone()); self.selected_text = row.full.clone(); }
                                            row_rect = Some(row_rect.map_or(r.rect, |rr| rr.union(r.rect)));
                                        });
                                        trow.col(|ui| {
                                            let mut rt = egui::RichText::new(opt_fmt(row.tv)).monospace();
                                            if selected { rt = rt.strong(); }
                                            let mut lbl = egui::Label::new(rt).sense(egui::Sense::click());
                                            lbl = lbl.wrap(false).truncate(true);
                                            let r = ui.add(lbl);
                                            if r.clicked() { self.selected_cid = Some(row.cid.clone()); self.selected_text = row.full.clone(); }
                                            row_rect = Some(row_rect.map_or(r.rect, |rr| rr.union(r.rect)));
                                        });
                                        trow.col(|ui| {
                                            let mut rt = egui::RichText::new(opt_fmt(row.vec)).monospace();
                                            if selected { rt = rt.strong(); }
                                            let mut lbl = egui::Label::new(rt).sense(egui::Sense::click());
                                            lbl = lbl.wrap(false).truncate(true);
                                            let r = ui.add(lbl);
                                            if r.clicked() { self.selected_cid = Some(row.cid.clone()); self.selected_text = row.full.clone(); }
                                            row_rect = Some(row_rect.map_or(r.rect, |rr| rr.union(r.rect)));
                                        });
                                        trow.col(|ui| {
                                            let mut rt = egui::RichText::new(format!("{:.4}", row.combined)).monospace();
                                            if selected { rt = rt.strong(); }
                                            let mut lbl = egui::Label::new(rt).sense(egui::Sense::click());
                                            lbl = lbl.wrap(false).truncate(true);
                                            let r = ui.add(lbl);
                                            if r.clicked() { self.selected_cid = Some(row.cid.clone()); self.selected_text = row.full.clone(); }
                                            row_rect = Some(row_rect.map_or(r.rect, |rr| rr.union(r.rect)));
                                        });
                                        trow.col(|ui| {
                                            let mut rt = egui::RichText::new(row.preview.clone());
                                            if selected { rt = rt.strong(); }
                                            let r = ui.add(
                                                egui::Label::new(rt)
                                                    .wrap(false)
                                                    .truncate(true)
                                                    .sense(egui::Sense::click()),
                                            );
                                            if r.clicked() { self.selected_cid = Some(row.cid.clone()); self.selected_text = row.full.clone(); }
                                            row_rect = Some(row_rect.map_or(r.rect, |rr| rr.union(r.rect)));
                                            if let Some(rr) = row_rect {
                                                // Draw a crisp separator slightly above the bottom edge
                                                let y = rr.max.y - 0.5;
                                                let a = egui::pos2(rr.min.x, y);
                                                let b = egui::pos2(rr.max.x, y);
                                                ui.painter().line_segment([a, b], egui::Stroke::new(1.0, egui::Color32::from_gray(230)));
                                            }
                                        });
                                    });
                                    }
                                });
                        });

                        ui.separator();
                        ui.heading("Full Text");
                        let hint = if let Some(cid) = &self.selected_cid { cid.as_str() } else { "(select a row)" };
                        ui.label(format!("Chunk: {}", hint));
                        ScrollArea::vertical().id_source("full_text_scroll").max_height(260.0).auto_shrink([false;2]).show(ui, |ui| {
                            ui.monospace(&self.selected_text);
                        });
                    }
                    ActiveTab::ExcelIngest => {
                        ui.heading("Ingest from Excel (first sheet, first column)");
                        ui.horizontal(|ui| {
                            ui.label("Input workbook:");
                            ui.add(TextEdit::singleline(&mut self.input_excel_path).desired_width(420.0));
                            if ui.add(Button::new("Browse")).clicked() {
                                if let Some(path) = FileDialog::new().add_filter("Excel", &["xlsx", "xls"]).pick_file() {
                                    self.input_excel_path = path.display().to_string();
                                }
                            }
                        });
                        ui.horizontal(|ui| { ui.checkbox(&mut self.excel_skip_header, "Skip first row (header)"); });
                        ui.horizontal(|ui| {
                            ui.label("Batch size:");
                            ui.add(egui::Slider::new(&mut self.excel_batch_size, 1..=512).text("rows/batch"));
                            let run_btn = ui.button("Run Ingest");
                            if run_btn.clicked() {
                                if self.ensure_embedder_or_queue(PendingAction::Insert) { self.run_excel_ingest(); }
                            }
                        });
                    }
                }
            });
        });
    }
}

// --- helpers ---
fn derive_hnsw_dir(db_path: &str) -> String { format!("{}.hnsw", db_path) }
#[cfg(feature = "tantivy")]
fn derive_tantivy_dir(db_path: &str) -> String { format!("{}.tantivy", db_path) }

fn make_ids_from_text(doc_hint: &str, text: &str) -> (DocumentId, ChunkId) {
    if !doc_hint.trim().is_empty() { return (DocumentId(doc_hint.to_string()), ChunkId(format!("{}#0", doc_hint))); }
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    let h = hasher.finish();
    let ts = Utc::now().timestamp_millis();
    let doc_id = format!("doc-{ts:x}-{h:08x}");
    let chunk_id = format!("{}#0", doc_id);
    (DocumentId(doc_id), ChunkId(chunk_id))
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

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if max_chars == 0 { return String::new(); }
    let mut it = s.chars();
    let truncated: String = it.by_ref().take(max_chars).collect();
    if it.next().is_some() {
        // There are more chars beyond the limit; append ellipsis.
        format!("{}…", truncated)
    } else {
        truncated
    }
}

#[derive(Debug, Clone, Default)]
struct HitRow {
    cid: String,
    fts: Option<f32>,
    tv: Option<f32>,
    vec: Option<f32>,
    combined: f32,
    preview: String,
    full: String,
}
impl HitRow { fn empty() -> Self { Self::default() } }

fn opt_fmt(v: Option<f32>) -> String {
    match v { Some(x) => format!("{:.4}", x), None => "-".into() }
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
        text: text.to_string(),
        section_path: Vec::new(),
        meta: BTreeMap::new(),
        extra: BTreeMap::new(),
    }
}
