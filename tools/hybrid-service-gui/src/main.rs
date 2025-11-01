use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::{mpsc::{self, Receiver, TryRecvError}, Arc};
use std::time::Instant;

use eframe::egui::{self, Button, CentralPanel, ComboBox, ScrollArea, Spinner, TextEdit, DragValue};
use eframe::egui::ProgressBar;
use egui_extras::{Column, TableBuilder, StripBuilder, Size};
use eframe::{App, CreationContext, Frame, NativeOptions};
use rfd::FileDialog;
use serde::{Deserialize, Serialize};

use hybrid_service::{HybridService, ServiceConfig, CancelToken, ProgressEvent, HnswState};
use embedding_provider::config::{default_stdio_config, ONNX_STDIO_DEFAULTS};
use chunking_store::{FilterClause, FilterKind, FilterOp};
use chunking_store::ChunkStoreRead;
// Removed unused FilterKind/FilterOp after moving Tantivy ops into service
#[cfg(feature = "tantivy")]
use chunking_store::tantivy_index::TantivyIndex;
use chunk_model::{ChunkId, DocumentId, ChunkRecord, FileRecord, SCHEMA_MAJOR};

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
    Files,
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

    // Store/index config (root -> derive artifacts)
    store_root: String,
    db_path: String,
    hnsw_dir: String,
    #[cfg(feature = "tantivy")]
    tantivy_dir: String,
    #[cfg(feature = "tantivy")]
    tantivy: Option<TantivyIndex>,
    #[cfg(feature = "tantivy")]
    last_tantivy_dir_applied: Option<String>,

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
    // Weights for result fusion (Hybrid)
    // None means: treat as not provided (null) and hide corresponding score column in results
    w_tv: Option<f32>,
    w_tv_and: Option<f32>,
    w_tv_or: Option<f32>,
    w_vec: Option<f32>,
    results: Vec<HitRow>,
    search_mode: SearchMode,

    // UI
    tab: ActiveTab,
    insert_mode: InsertMode,
    status: String,
    selected_cid: Option<String>,
    selected_text: String,
    selected_display: String,
    // Dangerous actions confirmation
    delete_confirm: String,

    // Preview Chunks popup
    preview_visible: bool,
    preview_chunks: Vec<ChunkRecord>,
    preview_selected: Option<usize>,
    preview_show_tab_escape: bool,

    // ONNX Runtime DLL lock (set after first successful Init)
    ort_runtime_committed: Option<String>,

    // Suggested filename for config save dialog
    config_last_name: String,
    // Optional store name to include in suggested config filename
    config_store_name: String,

    // Track last applied Store Root to auto-apply on Search/Insert
    last_store_root_applied: Option<String>,

    // Validation message for Store Root (shown under the field on failure)
    store_root_error: String,

    // Files tab
    files: Vec<FileRecord>,
    files_loading: bool,
    files_page: usize,
    files_page_size: usize,
    files_selected_doc: Option<String>,
    files_selected_display: String,
    files_selected_detail: String,
    files_delete_pending: Option<String>,
    files_deleting: bool,
}

// New (nested) config format: { store: {...}, chunk: {...}, model: {...} }
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HybridGuiConfigV2 {
    store: StoreCfg,
    chunk: ChunkCfg,
    model: ModelCfg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoreCfg {
    store_root: String,
    #[serde(default)]
    store_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChunkCfg {
    chunk_min: usize,
    chunk_max: usize,
    chunk_cap: usize,
    chunk_penalize_short_line: bool,
    chunk_penalize_page_no_nl: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelCfg {
    model_path: String,
    tokenizer_path: String,
    runtime_path: String,
    embedding_dimension: usize,
    max_tokens: usize,
    embed_batch_size: usize,
    embed_auto: bool,
    embed_initial_batch: usize,
    embed_min_batch: usize,
}

// Backward-compatible (flat) config format used previously
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HybridGuiConfigV1 {
    // Store
    store_root: String,
    // Chunking params
    chunk_min: usize,
    chunk_max: usize,
    chunk_cap: usize,
    chunk_penalize_short_line: bool,
    chunk_penalize_page_no_nl: bool,
    // Embed/model/runtime
    model_path: String,
    tokenizer_path: String,
    runtime_path: String,
    embedding_dimension: usize,
    max_tokens: usize,
    embed_batch_size: usize,
    embed_auto: bool,
    embed_initial_batch: usize,
    embed_min_batch: usize,
}

impl AppState {
    fn ui_files(&mut self, ui: &mut egui::Ui) {
        ui.heading("Files");
        if self.files.is_empty() && !self.files_loading {
            // Lazy-load first page on initial open
            self.refresh_files();
        }
        // Controls row
        ui.horizontal(|ui| {
            if ui.add(Button::new("Refresh")).clicked() {
                self.refresh_files();
            }
            ui.label("Page size");
            ui.add(DragValue::new(&mut self.files_page_size).clamp_range(5..=200));
            if ui.add(Button::new("Prev")).clicked() {
                if self.files_page > 0 { self.files_page -= 1; self.refresh_files(); }
            }
            if ui.add(Button::new("Next")).clicked() {
                // Optimistic paging; disable if last fetch was shorter than page size
                if self.files.len() >= self.files_page_size { self.files_page += 1; self.refresh_files(); }
            }
            if self.files_loading { ui.add(Spinner::new()); }
        });

        ui.separator();
        // Table
        let mut to_delete_doc: Option<String> = None;
        ui.push_id("files_table", |ui| {
            egui::ScrollArea::horizontal().id_source("files_table_h").show(ui, |ui| {
            let mut table = TableBuilder::new(ui)
                .striped(true)
                .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                .column(Column::initial(260.0))   // doc id
                .column(Column::initial(320.0))   // source uri
                .column(Column::initial(120.0))   // mime
                .column(Column::initial(70.0))    // pages
                .column(Column::initial(70.0))    // chunks
                .column(Column::initial(170.0))   // extracted at
                .column(Column::initial(90.0));   // actions

            table
                .header(20.0, |mut header| {
                    header.col(|ui| { ui.label("Doc ID"); });
                    header.col(|ui| { ui.label("File"); });
                    header.col(|ui| { ui.label("MIME"); });
                    header.col(|ui| { ui.label("Pages"); });
                    header.col(|ui| { ui.label("Chunks"); });
                    header.col(|ui| { ui.label("Extracted"); });
                    header.col(|ui| { ui.label("Actions"); });
                })
                .body(|mut body| {
                    fn trunc(s: &str, n: usize) -> String {
                        if s.chars().count() <= n { return s.to_string(); }
                        let mut out: String = s.chars().take(n).collect();
                        out.push('…');
                        out
                    }
                    for rec in &self.files {
                        body.row(22.0, |mut row_ui| {
                            let doc_disp = trunc(&rec.doc_id.0, 42);
                            row_ui.col(|ui| {
                                if ui.link(doc_disp).clicked() {
                                    self.files_selected_doc = Some(rec.doc_id.0.clone());
                                    self.files_selected_display = rec.doc_id.0.clone();
                                    self.files_selected_detail = serde_json::to_string_pretty(rec).unwrap_or_else(|_| "<render error>".into());
                                }
                            });
                            let file_disp = trunc(&rec.source_uri, 60);
                            row_ui.col(|ui| {
                                if ui.link(file_disp).clicked() {
                                    self.files_selected_doc = Some(rec.doc_id.0.clone());
                                    self.files_selected_display = rec.source_uri.clone();
                                    self.files_selected_detail = serde_json::to_string_pretty(rec).unwrap_or_else(|_| "<render error>".into());
                                }
                            });
                            row_ui.col(|ui| { ui.label(&rec.source_mime); });
                            row_ui.col(|ui| { ui.label(rec.page_count.map(|v| v.to_string()).unwrap_or_else(|| "-".into())); });
                            row_ui.col(|ui| { ui.label(rec.chunk_count.map(|v| v.to_string()).unwrap_or_else(|| "-".into())); });
                            row_ui.col(|ui| { ui.label(&rec.extracted_at); });
                            // Actions
                            row_ui.col(|ui| {
                                let pending = self.files_delete_pending.as_ref().map(|s| s == &rec.doc_id.0).unwrap_or(false);
                                if !pending {
                                    let btn = egui::RichText::new("Delete").color(egui::Color32::RED);
                                    if ui.add_enabled(!self.files_deleting, Button::new(btn)).clicked() {
                                        self.files_delete_pending = Some(rec.doc_id.0.clone());
                                    }
                                } else {
                                    ui.horizontal(|ui| {
                                        let btn = egui::RichText::new("Confirm").color(egui::Color32::RED);
                                        if ui.add_enabled(!self.files_deleting, Button::new(btn)).clicked() {
                                            to_delete_doc = Some(rec.doc_id.0.clone());
                                        }
                                        if ui.button("Cancel").clicked() { self.files_delete_pending = None; }
                                    });
                                }
                            });
                        });
                    }
                });
            });
        });

        if let Some(doc) = to_delete_doc.take() {
            self.delete_by_doc_id(&doc);
        }

        if let Some(_doc) = &self.files_selected_doc {
            ui.separator();
            if self.files_selected_display.is_empty() {
                ui.label("Selected:");
            } else {
                ui.label(format!("Selected: {}", self.files_selected_display));
            }
            ScrollArea::vertical().max_height(220.0).id_source("files_selected_scroll").show(ui, |ui| {
                ui.add(TextEdit::multiline(&mut self.files_selected_detail).desired_rows(8).desired_width(800.0).id_source("files_selected_detail"));
            });
        }
    }

    fn delete_by_doc_id(&mut self, doc_id: &str) {
        if self.svc.is_none() { self.status = "Service not initialized".into(); return; }
        if !self.ensure_store_paths_current() { return; }
        let filters = vec![FilterClause { kind: FilterKind::Must, op: FilterOp::DocIdEq(doc_id.to_string()) }];
        self.files_deleting = true;
        if let Some(svc) = &self.svc {
            match svc.delete_by_filter(&filters, 1000) {
                Ok(rep) => {
                    self.status = format!("Deleted: db={} ids={}, batches={}", rep.db_deleted, rep.total_ids, rep.batches);
                    self.files_delete_pending = None;
                    self.files_selected_doc = None;
                    self.files_selected_display.clear();
                    self.files_selected_detail.clear();
                    // Refresh current page
                    self.refresh_files();
                }
                Err(e) => { self.status = format!("Delete failed: {e}"); }
            }
        }
        self.files_deleting = false;
    }

    fn refresh_files(&mut self) {
        if self.svc.is_none() { self.status = "Service not initialized".into(); return; }
        if !self.ensure_store_paths_current() { return; }
        let offset = self.files_page.saturating_mul(self.files_page_size);
        let limit = self.files_page_size;
        self.files_loading = true;
        // Synchronous fetch; can move to a thread if needed later
        if let Some(svc) = &self.svc {
            match svc.list_files(limit, offset) {
                Ok(list) => { self.files = list; self.status = format!("Loaded files page {} ({} items)", self.files_page + 1, self.files.len()); }
                Err(e) => { self.status = format!("List files failed: {e}"); }
            }
        }
        self.files_loading = false;
    }
    fn apply_store_root_now(&mut self, reason: &str) {
        // Take an owned copy to avoid borrowing self across mutable calls
        let root = self.store_root.trim().to_string();
        let p = std::path::Path::new(root.as_str());
        if !p.exists() || !p.is_dir() {
            self.store_root_error = format!("Invalid Store Root (not an existing directory): {}", root);
            return;
        }
        self.store_root_error.clear();
        self.refresh_store_paths();
        std::env::set_var("HYBRID_STORE_ROOT", root.as_str());
        // Create derived subdirs if missing, but do not create the root here
        let _ = fs::create_dir_all(derive_hnsw_dir(root.as_str()));
        #[cfg(feature = "tantivy")] let _ = fs::create_dir_all(derive_tantivy_dir(root.as_str()));
        #[cfg(feature = "tantivy")] { self.tantivy = None; }
        if let Some(svc) = &self.svc {
            svc.set_store_paths(PathBuf::from(self.db_path.trim()), Some(PathBuf::from(self.hnsw_dir.trim())));
        }
        self.last_store_root_applied = Some(root.clone());
        self.status = format!("{}: {}", reason, root);
    }
    fn ui_config(&mut self, ui: &mut egui::Ui) {
        ui.heading("Model / Store Config");
        ui.add_enabled_ui(!self.ingest_running, |ui| {
            // Config load/save row
            ui.horizontal(|ui| {
                if ui.button("Load Config").clicked() { self.load_config_via_dialog(); }
                if ui.button("Save Config").clicked() { self.save_config_via_dialog(); }
            });
            ui.horizontal(|ui| {
                ui.label("Store Name (Optional)");
                ui.add(TextEdit::singleline(&mut self.config_store_name).desired_width(200.0));
            });
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Store Root");
                if ui.button("Browse").clicked() {
                    if let Some(p) = FileDialog::new().pick_folder() {
                        self.store_root = p.display().to_string();
                        self.apply_store_root_now("Store root set via Browse");
                    }
                }
                let resp = ui.add(TextEdit::singleline(&mut self.store_root).desired_width(400.0));
                let commit_enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                if resp.lost_focus() || commit_enter {
                    self.apply_store_root_now("Store root applied");
                }
            });
            if !self.store_root_error.is_empty() {
                ui.label(egui::RichText::new(&self.store_root_error).color(ui.visuals().warn_fg_color));
            }
            ui.horizontal(|ui| { ui.label("DB"); ui.label(&self.db_path); });
            ui.horizontal(|ui| { ui.label("HNSW"); ui.label(&self.hnsw_dir); });
            #[cfg(feature = "tantivy")]
            ui.horizontal(|ui| { ui.label("Tantivy"); ui.label(&self.tantivy_dir); });
            // Danger zone (always visible, requires typing 'Activate' and pressing 'Delete')
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Danger zone").color(egui::Color32::LIGHT_RED));
                ui.label("Input \"Activate\" and Push \"Delete\" to Delete DB and Indexes");
                ui.add(TextEdit::singleline(&mut self.delete_confirm).desired_width(140.0));
                let enabled = self.delete_confirm.trim() == "Activate";
                let btn = egui::RichText::new("Delete").color(egui::Color32::RED);
                if ui.add_enabled(enabled, Button::new(btn)).clicked() {
                    self.delete_store_files();
                    self.delete_confirm.clear();
                }
            });
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
            // (moved Danger zone above Chunking Params)
            ui.separator();
            ui.horizontal(|ui| { ui.label("Model"); ui.add(TextEdit::singleline(&mut self.model_path).desired_width(400.0)); if ui.button("Browse").clicked() { if let Some(p) = FileDialog::new().add_filter("ONNX", &["onnx"]).pick_file() { self.model_path = p.display().to_string(); } } });
            ui.horizontal(|ui| { ui.label("Tokenizer"); ui.add(TextEdit::singleline(&mut self.tokenizer_path).desired_width(400.0)); if ui.button("Browse").clicked() { if let Some(p) = FileDialog::new().add_filter("JSON", &["json"]).pick_file() { self.tokenizer_path = p.display().to_string(); } } });
            ui.horizontal(|ui| {
                ui.label("Runtime DLL");
                ui.add(TextEdit::singleline(&mut self.runtime_path).desired_width(400.0));
                if ui.button("Browse").clicked() { if let Some(p) = FileDialog::new().pick_file() { self.runtime_path = p.display().to_string(); } }
            });
            let msg = "After the first Init, the Runtime DLL cannot be changed within this session. Restart the app to apply a different DLL.";
            ui.label(egui::RichText::new(msg).color(ui.visuals().warn_fg_color));
            ui.horizontal(|ui| {
                ui.label("Dim"); ui.add(TextEdit::singleline(&mut self.embedding_dimension).desired_width(80.0));
                ui.label("MaxTokens"); ui.add(TextEdit::singleline(&mut self.max_tokens).desired_width(80.0));
                ui.label("Batch"); ui.add(TextEdit::singleline(&mut self.embed_batch_size).desired_width(60.0));
            });
            ui.horizontal(|ui| {
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
    
    fn to_ui_config_v2(&self) -> HybridGuiConfigV2 {
        HybridGuiConfigV2 {
            store: StoreCfg {
                store_root: self.store_root.trim().to_string(),
                store_name: {
                    let n = self.config_store_name.trim();
                    if n.is_empty() { None } else { Some(n.to_string()) }
                },
            },
            chunk: ChunkCfg {
                chunk_min: self.chunk_min.trim().parse().unwrap_or(400),
                chunk_max: self.chunk_max.trim().parse().unwrap_or(600),
                chunk_cap: self.chunk_cap.trim().parse().unwrap_or(800),
                chunk_penalize_short_line: self.chunk_penalize_short_line,
                chunk_penalize_page_no_nl: self.chunk_penalize_page_no_nl,
            },
            model: ModelCfg {
                model_path: self.model_path.trim().to_string(),
                tokenizer_path: self.tokenizer_path.trim().to_string(),
                runtime_path: self.runtime_path.trim().to_string(),
                embedding_dimension: self.embedding_dimension.trim().parse().unwrap_or_default(),
                max_tokens: self.max_tokens.trim().parse().unwrap_or_default(),
                embed_batch_size: self.embed_batch_size.trim().parse().unwrap_or(64),
                embed_auto: self.embed_auto,
                embed_initial_batch: self.embed_initial_batch.trim().parse().unwrap_or(128),
                embed_min_batch: self.embed_min_batch.trim().parse().unwrap_or(8),
            },
        }
    }

    fn apply_ui_config_v2(&mut self, cfg: HybridGuiConfigV2) {
        // Store
        self.store_root = cfg.store.store_root;
        self.refresh_store_paths();
        std::env::set_var("HYBRID_STORE_ROOT", self.store_root.trim());
        #[cfg(feature = "tantivy")]
        { self.tantivy = None; }
        self.config_store_name = cfg.store.store_name.unwrap_or_default();
        // Chunking params
        self.chunk_min = cfg.chunk.chunk_min.to_string();
        self.chunk_max = cfg.chunk.chunk_max.to_string();
        self.chunk_cap = cfg.chunk.chunk_cap.to_string();
        self.chunk_penalize_short_line = cfg.chunk.chunk_penalize_short_line;
        self.chunk_penalize_page_no_nl = cfg.chunk.chunk_penalize_page_no_nl;
        // Model
        self.model_path = cfg.model.model_path;
        self.tokenizer_path = cfg.model.tokenizer_path;
        self.runtime_path = cfg.model.runtime_path;
        self.embedding_dimension = cfg.model.embedding_dimension.to_string();
        self.max_tokens = cfg.model.max_tokens.to_string();
        self.embed_batch_size = cfg.model.embed_batch_size.to_string();
        self.embed_auto = cfg.model.embed_auto;
        self.embed_initial_batch = cfg.model.embed_initial_batch.to_string();
        self.embed_min_batch = cfg.model.embed_min_batch.to_string();
    }

    fn apply_ui_config_v1(&mut self, cfg: HybridGuiConfigV1) {
        // Store
        self.store_root = cfg.store_root;
        self.refresh_store_paths();
        // Chunking params
        self.chunk_min = cfg.chunk_min.to_string();
        self.chunk_max = cfg.chunk_max.to_string();
        self.chunk_cap = cfg.chunk_cap.to_string();
        self.chunk_penalize_short_line = cfg.chunk_penalize_short_line;
        self.chunk_penalize_page_no_nl = cfg.chunk_penalize_page_no_nl;
        // Model
        self.model_path = cfg.model_path;
        self.tokenizer_path = cfg.tokenizer_path;
        self.runtime_path = cfg.runtime_path;
        self.embedding_dimension = cfg.embedding_dimension.to_string();
        self.max_tokens = cfg.max_tokens.to_string();
        self.embed_batch_size = cfg.embed_batch_size.to_string();
        self.embed_auto = cfg.embed_auto;
        self.embed_initial_batch = cfg.embed_initial_batch.to_string();
        self.embed_min_batch = cfg.embed_min_batch.to_string();
    }

    fn load_config_via_dialog(&mut self) {
        if let Some(path) = FileDialog::new().add_filter("JSON", &["json"]).pick_file() {
            match std::fs::read_to_string(&path) {
                Ok(s) => {
                    // Try V2 first, then V1 for backward compatibility
                    if let Ok(cfg2) = serde_json::from_str::<HybridGuiConfigV2>(&s) {
                        self.apply_ui_config_v2(cfg2);
                        self.status = format!("Loaded config (v2) from {}", path.display());
                        if let Some(name) = std::path::Path::new(&path).file_name().and_then(|s| s.to_str()) { self.config_last_name = name.to_string(); }
                    } else if let Ok(cfg1) = serde_json::from_str::<HybridGuiConfigV1>(&s) {
                        self.apply_ui_config_v1(cfg1);
                        self.status = format!("Loaded config (v1) from {}", path.display());
                        if let Some(name) = std::path::Path::new(&path).file_name().and_then(|s| s.to_str()) { self.config_last_name = name.to_string(); }
                    } else {
                        self.status = format!("Load config failed: invalid JSON structure");
                    }
                }
                Err(e) => { self.status = format!("Load config failed: {}", e); }
            }
        }
    }

    fn save_config_via_dialog(&mut self) {
        let suggested = self.suggest_config_filename();
        if let Some(path) = FileDialog::new().add_filter("JSON", &["json"]).set_file_name(&suggested).save_file() {
            let cfg = self.to_ui_config_v2();
            match serde_json::to_string_pretty(&cfg) {
                Ok(body) => match std::fs::write(&path, body) {
                    Ok(_) => {
                        self.status = format!("Saved config to {}", path.display());
                        if let Some(name) = std::path::Path::new(&path).file_name().and_then(|s| s.to_str()) {
                            self.config_last_name = name.to_string();
                        }
                    }
                    Err(e) => { self.status = format!("Save config failed: {}", e); }
                }
                Err(e) => { self.status = format!("Serialize config failed: {}", e); }
            }
        }
    }

    fn suggest_config_filename(&self) -> String {
        let base = "hybrid-service-gui";
        let name = self.config_store_name.trim();
        if name.is_empty() {
            format!("{}.json", base)
        } else {
            let safe = safe_filename_component(name);
            format!("{}.{}.json", base, safe)
        }
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
        let need_reopen = match &self.last_tantivy_dir_applied {
            Some(applied) => applied != &self.tantivy_dir,
            None => true,
        };
        if self.tantivy.is_none() || need_reopen {
            let dir = &self.tantivy_dir;
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
            let idx = TantivyIndex::open_or_create_dir(dir).map_err(|e| e.to_string())?;
            self.tantivy = Some(idx);
            self.last_tantivy_dir_applied = Some(self.tantivy_dir.clone());
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

            store_root: store_default.clone(),
            db_path: derive_db_path(&store_default),
            hnsw_dir: derive_hnsw_dir(&store_default),
            #[cfg(feature = "tantivy")]
            tantivy_dir: derive_tantivy_dir(&store_default),
            #[cfg(feature = "tantivy")]
            tantivy: None,
            #[cfg(feature = "tantivy")]
            last_tantivy_dir_applied: None,

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
            // Default to OR/VEC enabled (1.0) to match previous behavior; TV/AND left unset
            w_tv: None,
            w_tv_and: None,
            w_tv_or: Some(1.0),
            w_vec: Some(1.0),
            results: Vec::new(),
            search_mode: SearchMode::Hybrid,

            tab: ActiveTab::Insert,
            insert_mode: InsertMode::File,
            status: String::new(),
            selected_cid: None,
            selected_text: String::new(),
            selected_display: String::new(),
            delete_confirm: String::new(),

            preview_visible: false,
            preview_chunks: Vec::new(),
            preview_selected: None,
            preview_show_tab_escape: true,

            ort_runtime_committed: None,

            config_last_name: String::from("config.json"),
            config_store_name: String::new(),

            last_store_root_applied: None,

            store_root_error: String::new(),

            // Files tab
            files: Vec::new(),
            files_loading: false,
            files_page: 0,
            files_page_size: 20,
            files_selected_doc: None,
            files_selected_display: String::new(),
            files_selected_detail: String::new(),
            files_delete_pending: None,
            files_deleting: false,
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
        // Use default preload behavior from embedder config (no GUI override)
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

    // Ensure that DB/HNSW paths reflect the current Store Root before operations.
    fn ensure_store_paths_current(&mut self) -> bool {
        let current = self.store_root.trim().to_string();
        let needs_apply = match &self.last_store_root_applied { Some(prev) => prev != &current, None => true };
        if !needs_apply { return true; }
        // Validate existence first; do not auto-create root
        let p = std::path::Path::new(&current);
        if !p.exists() || !p.is_dir() {
            self.store_root_error = format!("Invalid Store Root (not an existing directory): {}", current);
            return false;
        }
        self.store_root_error.clear();
        self.refresh_store_paths();
        std::env::set_var("HYBRID_STORE_ROOT", current.as_str());
        let _ = fs::create_dir_all(derive_hnsw_dir(current.as_str()));
        #[cfg(feature = "tantivy")]
        let _ = fs::create_dir_all(derive_tantivy_dir(current.as_str()));
        #[cfg(feature = "tantivy")]
        { self.tantivy = None; }
        if let Some(svc) = &self.svc {
            svc.set_store_paths(PathBuf::from(self.db_path.trim()), Some(PathBuf::from(self.hnsw_dir.trim())));
        }
        self.last_store_root_applied = Some(current);
        true
    }

    fn poll_service_task(&mut self) {
        if let Some(task) = &self.svc_task {
            match task.rx.try_recv() {
                Ok(Ok(svc)) => {
                    self.svc = Some(svc);
                    self.status = format!("Model ready in {:.1}s", task.started.elapsed().as_secs_f32());
                    // Install provider that reads current store root from environment.
                    if let Some(svc) = &self.svc {
                        let provider = Arc::new(|| {
                            let root = std::env::var("HYBRID_STORE_ROOT").unwrap_or_else(|_| String::from("target/demo/store"));
                            let db = std::path::PathBuf::from(derive_db_path(root.as_str()));
                            let hnsw = Some(std::path::PathBuf::from(derive_hnsw_dir(root.as_str())));
                            (db, hnsw)
                        });
                        svc.set_store_path_provider(provider);
                    }
                    // Seed env var right away
                    std::env::set_var("HYBRID_STORE_ROOT", self.store_root.trim());
                    if self.ort_runtime_committed.is_none() {
                        self.ort_runtime_committed = Some(self.runtime_path.trim().to_string());
                    }
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
                    // Top-level tabs with underline accent
                    let resp_insert = ui.selectable_value(&mut self.tab, ActiveTab::Insert, "Insert");
                    let resp_search = ui.selectable_value(&mut self.tab, ActiveTab::Search, "Search");
                    let resp_files = ui.selectable_value(&mut self.tab, ActiveTab::Files, "Files");
                    let resp_config = ui.selectable_value(&mut self.tab, ActiveTab::Config, "Config");
                    let union12 = resp_insert.rect.union(resp_search.rect);
                    let union123 = union12.union(resp_files.rect);
                    let union_all = union123.union(resp_config.rect);
                    let y = union_all.bottom() + 2.0;
                    let painter = ui.painter();
                    let base_color = ui.visuals().widgets.noninteractive.bg_stroke.color;
                    let accent = ui.visuals().selection.stroke.color;
                    // Baseline across all tabs
                    painter.line_segment(
                        [egui::pos2(union_all.left(), y), egui::pos2(union_all.right(), y)],
                        egui::Stroke { width: 1.0, color: base_color },
                    );
                    // Active tab underline
                    let active_rect = match self.tab {
                        ActiveTab::Insert => resp_insert.rect,
                        ActiveTab::Search => resp_search.rect,
                        ActiveTab::Files => resp_files.rect,
                        ActiveTab::Config => resp_config.rect,
                    };
                    painter.line_segment(
                        [egui::pos2(active_rect.left(), y), egui::pos2(active_rect.right(), y)],
                        egui::Stroke { width: 2.0, color: accent },
                    );
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
                    let dll_status = if self.ort_runtime_committed.is_some() { "fixed" } else { "editable" };
                    let status_text = format!("Model: {}, RuntimeDLL: {}, Index: {}", model_status, dll_status, index_status);
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
                ActiveTab::Files => self.ui_files(ui),
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
                // Draw two tab-like buttons with an underline
                let resp_file = ui.selectable_value(&mut self.insert_mode, InsertMode::File, "Insert File");
                let resp_text = ui.selectable_value(&mut self.insert_mode, InsertMode::Text, "Insert Text");

                // Compute a union rect spanning both buttons
                let union = resp_file.rect.union(resp_text.rect);
                let y = union.bottom() + 2.0;
                let painter = ui.painter();
                let base_color = ui.visuals().widgets.noninteractive.bg_stroke.color;
                let accent = ui.visuals().selection.stroke.color;

                // Baseline under both tabs
                painter.line_segment(
                    [egui::pos2(union.left(), y), egui::pos2(union.right(), y)],
                    egui::Stroke { width: 1.0, color: base_color },
                );

                // Accent underline under the active tab
                let active_rect = match self.insert_mode {
                    InsertMode::File => resp_file.rect,
                    InsertMode::Text => resp_text.rect,
                };
                painter.line_segment(
                    [egui::pos2(active_rect.left(), y), egui::pos2(active_rect.right(), y)],
                    egui::Stroke { width: 2.0, color: accent },
                );
            });
            ui.add_space(6.0);

            match self.insert_mode {
                InsertMode::File => {
                    // Row 1: Choose File, [File path]
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
                    });

                    // Row 2: Encoding selector (for text-like files)
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
                        });
                        // Row 3: Preview text (separate line)
                        let mut preview_short: String = self.ingest_preview.chars().take(60).collect();
                        if self.ingest_preview.chars().count() > 60 { preview_short.push('…'); }
                        ui.label(format!("Preview: {}", preview_short.replace(['\n','\r','\t'], " ")));
                    }

                    // Row 4: [Preview Chunks] button
                    if ui.add_enabled(!self.ingest_running, Button::new("Preview Chunks")).clicked() {
                        self.do_preview_chunks();
                    }

                    // Row 5: Ingest File
                    if ui.add_enabled(!self.ingest_running, Button::new("Ingest File")).clicked() { self.do_ingest_file(); }
                }
                InsertMode::Text => {
                    // Text input with the action button placed below
                    ui.label("Text");
                    ui.add(TextEdit::multiline(&mut self.input_text).desired_rows(4).desired_width(600.0));
                    if ui.add(Button::new("Insert Text")).clicked() { self.do_insert_text(); }
                }
            }
        });
        // Draw the preview popup on top if requested
        self.ui_preview_window(ui.ctx());
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

    fn do_preview_chunks(&mut self) {
        let path = self.ingest_file_path.trim();
        if path.is_empty() { self.status = "Pick a file to preview".into(); return; }
        // Build params from UI
        let mut params = file_chunker::text_segmenter::TextChunkParams::default();
        if let Ok(v) = self.chunk_min.trim().parse::<usize>() { if v > 0 { params.min_chars = v; } }
        if let Ok(v) = self.chunk_max.trim().parse::<usize>() { if v > 0 { params.max_chars = v; } }
        if let Ok(v) = self.chunk_cap.trim().parse::<usize>() { if v > 0 { params.cap_chars = v; } }
        params.penalize_short_line = self.chunk_penalize_short_line;
        params.penalize_page_boundary_no_newline = self.chunk_penalize_page_no_nl;

        // Encoding: None when auto, otherwise pass through
        let enc_lower = self.ingest_encoding.to_ascii_lowercase();
        let enc_opt: Option<String> = if enc_lower == "auto" { None } else { Some(enc_lower) };

        // Compute chunks using file-chunker with unified text params
        let out = file_chunker::chunk_file_with_file_record_with_params(path, enc_opt.as_deref(), &params);
        self.preview_chunks = out.chunks;
        self.preview_selected = None;
        self.preview_visible = true;
        self.status = format!("Preview loaded ({} chunks)", self.preview_chunks.len());
    }

    fn ui_preview_window(&mut self, ctx: &egui::Context) {
        if !self.preview_visible { return; }
        let mut open = self.preview_visible;
        let mut request_close = false;
        egui::Window::new("Preview Chunks")
            .open(&mut open)
            .collapsible(false)
            .default_width(840.0)
            .default_height(600.0)
            .default_pos(egui::pos2(40.0, 40.0))
            .show(ctx, |ui| {
                // Toolbar
                // Row 1: Close + checkbox
                ui.horizontal(|ui| {
                    let close_btn = Button::new(egui::RichText::new("Close Preview").color(egui::Color32::RED).strong());
                    if ui.add(close_btn).clicked() { request_close = true; }
                    ui.add_space(8.0);
                    ui.checkbox(&mut self.preview_show_tab_escape, "Show \\t for tabs");
                });
                // Row 2: File path
                ui.horizontal(|ui| {
                    ui.label(format!("File: {}", self.ingest_file_path.trim()));
                });
                ui.separator();
                // Keyboard: ESC to close
                if ui.input(|i| i.key_pressed(egui::Key::Escape)) { request_close = true; }

                // Two-row layout: list (top) / selected (bottom)
                StripBuilder::new(ui)
                    .size(Size::relative(0.55))
                    .size(Size::remainder())
                    .clip(true)
                    .vertical(|mut strip| {
                        // Top: chunks list
                        strip.cell(|ui| {
                            ui.heading(format!("Chunks ({}):", self.preview_chunks.len()));
                            ui.separator();
                            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                                for (i, c) in self.preview_chunks.iter().enumerate() {
                                    let preview = truncate_for_preview(&c.text, 80);
                                    let page_label = match (c.page_start, c.page_end) {
                                        (Some(s), Some(e)) if s == e => format!("{}", s),
                                        (Some(s), Some(e)) => format!("{}-{}", s, e),
                                        (Some(s), None) => format!("{}", s),
                                        _ => String::new(),
                                    };
                                    let title = if page_label.is_empty() {
                                        format!("{}", preview)
                                    } else {
                                        format!("#{}  {}", page_label, preview)
                                    };
                                    let click = ui.selectable_label(self.preview_selected == Some(i), title);
                                    if click.clicked() { self.preview_selected = Some(i); }
                                }
                            });
                        });

                        // Bottom: selected chunk detail
                        strip.cell(|ui| {
                            egui::Frame::default().show(ui, |ui| {
                                ui.heading("Selected Chunk");
                                ui.separator();
                                if let Some(i) = self.preview_selected { if let Some(c) = self.preview_chunks.get(i) {
                                    let text = if self.preview_show_tab_escape { escape_tabs(&c.text) } else { c.text.clone() };
                                    let page_label = match (c.page_start, c.page_end) {
                                        (Some(s), Some(e)) if s == e => format!("{}", s),
                                        (Some(s), Some(e)) => format!("{}-{}", s, e),
                                        (Some(s), None) => format!("{}", s),
                                        _ => String::new(),
                                    };
                                    if page_label.is_empty() {
                                        ui.monospace(format!("len={} bytes", c.text.len()));
                                    } else {
                                        ui.monospace(format!("len={} bytes  |  {}", c.text.len(), page_label));
                                    }
                                    ui.separator();
                                    egui::ScrollArea::vertical().show(ui, |ui| { ui.monospace(text); });
                                }}
                            });
                        });
                    });
            });
        if request_close { open = false; }
        self.preview_visible = open;
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
        // Auto-apply Store Root before writing into the DB via service
        if !self.ensure_store_paths_current() { return; }
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
        // Auto-apply Store Root before ingesting
        if !self.ensure_store_paths_current() { return; }
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
                                // Tantivy upsert is handled in the service now
                                self.ingest_doc_key = None;
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
            });

            // Row 3: Weights (under TopK/Mode), only in Hybrid mode
            if matches!(self.search_mode, SearchMode::Hybrid) {
                ui.horizontal(|ui| {
                    // Four weights (nullable). Empty input => None (hide column). 0 is valid.
                    let mut tv_s = self.w_tv.map(|v| format!("{}", v)).unwrap_or_default();
                    let mut tv_and_s = self.w_tv_and.map(|v| format!("{}", v)).unwrap_or_default();
                    let mut tv_or_s = self.w_tv_or.map(|v| format!("{}", v)).unwrap_or_default();
                    let mut vec_s = self.w_vec.map(|v| format!("{}", v)).unwrap_or_default();

                    ui.label("w_TV");
                    if ui.add(TextEdit::singleline(&mut tv_s).desired_width(60.0).id_source("w_tv")).changed() {
                        let t = tv_s.trim(); self.w_tv = if t.is_empty() { None } else { t.parse::<f32>().ok() };
                    }
                    ui.label("w_TV(AND)");
                    if ui.add(TextEdit::singleline(&mut tv_and_s).desired_width(60.0).id_source("w_tv_and")).changed() {
                        let t = tv_and_s.trim(); self.w_tv_and = if t.is_empty() { None } else { t.parse::<f32>().ok() };
                    }
                    ui.label("w_TV(OR)");
                    if ui.add(TextEdit::singleline(&mut tv_or_s).desired_width(60.0).id_source("w_tv_or")).changed() {
                        let t = tv_or_s.trim(); self.w_tv_or = if t.is_empty() { None } else { t.parse::<f32>().ok() };
                    }
                    ui.label("w_VEC");
                    if ui.add(TextEdit::singleline(&mut vec_s).desired_width(60.0).id_source("w_vec")).changed() {
                        let t = vec_s.trim(); self.w_vec = if t.is_empty() { None } else { t.parse::<f32>().ok() };
                    }

                    // Show normalized percentages across the provided weights
                    let wt = self.w_tv.unwrap_or(0.0);
                    let wa = self.w_tv_and.unwrap_or(0.0);
                    let wo = self.w_tv_or.unwrap_or(0.0);
                    let wv = self.w_vec.unwrap_or(0.0);
                    let denom = (wt + wa + wo + wv).max(1e-6);
                    ui.label(format!("% TV:{:.0} AND:{:.0} OR:{:.0} VEC:{:.0}", (wt/denom)*100.0, (wa/denom)*100.0, (wo/denom)*100.0, (wv/denom)*100.0));
                });
            }

            ui.separator();
            ui.push_id("results_table", |ui| {
                egui::ScrollArea::horizontal().id_source("results_table_h").show(ui, |ui| {
                    let show_tv = self.w_tv.is_some();
                    let show_tv_and = self.w_tv_and.is_some();
                    let show_tv_or = self.w_tv_or.is_some();
                    let show_vec = self.w_vec.is_some();

                    let mut table = TableBuilder::new(ui)
                        .striped(true)
                        .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                        .column(Column::initial(36.0).at_least(30.0))
                        .column(Column::initial(220.0))   // file
                        .column(Column::initial(80.0));    // page

                    if show_tv { table = table.column(Column::initial(70.0)); }
                    if show_tv_and { table = table.column(Column::initial(80.0)); }
                    if show_tv_or { table = table.column(Column::initial(80.0)); }
                    if show_vec { table = table.column(Column::initial(70.0)); }

                    table = table.column(Column::remainder()); // text (after scores)

                    table
                        .header(20.0, |mut header| {
                            header.col(|ui| { ui.label("#"); });
                            header.col(|ui| { ui.label("File"); });
                            header.col(|ui| { ui.label("Page"); });
                            if show_tv { header.col(|ui| { ui.label("TV"); }); }
                            if show_tv_and { header.col(|ui| { ui.label("TV(AND)"); }); }
                            if show_tv_or { header.col(|ui| { ui.label("TV(OR)"); }); }
                            if show_vec { header.col(|ui| { ui.label("VEC"); }); }
                            header.col(|ui| { ui.label("Text"); });
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
                                    if show_tv { row_ui.col(|ui| { ui.label(opt_fmt(row.tv)); }); }
                                    if show_tv_and { row_ui.col(|ui| { ui.label(opt_fmt(row.tv_and)); }); }
                                    if show_tv_or { row_ui.col(|ui| { ui.label(opt_fmt(row.tv_or)); }); }
                                    if show_vec { row_ui.col(|ui| { ui.label(opt_fmt(row.vec)); }); }
                                    row_ui.col(|ui| {
                                        ui.push_id(i, |ui| {
                                            if ui.link(&row.text_preview).clicked() {
                                                self.selected_cid = Some(row.cid.clone());
                                                self.selected_text = row.text_full.clone();
                                                self.selected_display = format!("{} {}", &row.file, if row.page.is_empty() { String::new() } else { row.page.clone() });
                                            }
                                        });
                                    });
                                });
                            }
                        });
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
        // Auto-apply Store Root if user edited
        if !self.ensure_store_paths_current() { return; }
        let q_owned = self.query.clone();
        let q = q_owned.trim();
        if q.is_empty() { self.status = "Enter query".into(); return; }
        let filters: &[FilterClause] = &[];
        let top_k = self.top_k;
        // Allow FTS-only search without model initialization
        // Build union of results from TV, TV(AND), TV(OR), VEC
        use chunking_store::sqlite_repo::SqliteRepo;
        let repo = match SqliteRepo::open(self.db_path.trim()) { Ok(r) => r, Err(e) => { self.status = format!("Open DB failed: {e}"); return; } };
        // FTS5 is not used for search ranking here; skip any FTS maintenance.

        // Tantivy queries (skip when mode = VEC); now routed via service
        #[cfg(feature = "tantivy")]
        let (tv, tv_and, tv_or) = if matches!(self.search_mode, SearchMode::Vec) {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            if let Some(svc) = &self.svc {
                match svc.tantivy_triple(q, top_k, filters) {
                    Ok(t) => t,
                    Err(e) => { self.status = format!("Tantivy search failed: {}", e); return; }
                }
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
        let wt_raw = self.w_tv.unwrap_or(0.0);
        let wa_raw = self.w_tv_and.unwrap_or(0.0);
        let wo_raw = self.w_tv_or.unwrap_or(0.0);
        let wv_raw = self.w_vec.unwrap_or(0.0);
        let denom = (wt_raw + wa_raw + wo_raw + wv_raw).max(1e-6);
        let wt = wt_raw / denom;
        let wa = wa_raw / denom;
        let wo = wo_raw / denom;
        let wv = wv_raw / denom;
        items.sort_by(|a, b| {
            let (tv_a, tv_and_a, tv_or_a, vec_a) = (a.1.0.unwrap_or(0.0), a.1.1.unwrap_or(0.0), a.1.2.unwrap_or(0.0), a.1.3.unwrap_or(0.0));
            let (tv_b, tv_and_b, tv_or_b, vec_b) = (b.1.0.unwrap_or(0.0), b.1.1.unwrap_or(0.0), b.1.2.unwrap_or(0.0), b.1.3.unwrap_or(0.0));
            let key_a = match self.search_mode {
                SearchMode::Hybrid => wt * tv_a + wa * tv_and_a + wo * tv_or_a + wv * vec_a,
                SearchMode::Tantivy => tv_a.max(tv_and_a).max(tv_or_a),
                SearchMode::Vec => vec_a,
            };
            let key_b = match self.search_mode {
                SearchMode::Hybrid => wt * tv_b + wa * tv_and_b + wo * tv_or_b + wv * vec_b,
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

fn truncate_for_preview(s: &str, max_chars: usize) -> String {
    if max_chars == 0 { return String::new(); }
    let mut it = s.chars();
    let truncated: String = it.by_ref().take(max_chars).collect();
    if it.next().is_some() { format!("{}…", truncated.replace(['\n', '\r', '\t'], " ")) } else { truncated.replace(['\n', '\r', '\t'], " ") }
}

fn escape_tabs(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch { '\t' => out.push_str("\\t"), '\r' => { /* skip */ }, _ => out.push(ch) }
    }
    out
}

// Create a safe filename component while preserving Unicode characters.
// Replaces forbidden characters (\\ / : * ? " < > |) and control chars with '-',
// converts whitespace to '-', collapses repeated '-', and trims leading/trailing '-'.
// Falls back to "store" if the result is empty.
fn safe_filename_component(name: &str) -> String {
    let forbidden: [char; 9] = ['\\', '/', ':', '*', '?', '"', '<', '>', '|'];
    let mut out = String::with_capacity(name.len());
    let mut last_dash = false;
    for ch in name.chars() {
        let mapped = if forbidden.contains(&ch) || ch.is_control() {
            '-'
        } else if ch.is_whitespace() {
            '-'
        } else {
            ch
        };
        if mapped == '-' {
            if !last_dash {
                out.push('-');
                last_dash = true;
            }
        } else {
            out.push(mapped);
            last_dash = false;
        }
    }
    let trimmed = out.trim_matches('-').to_string();
    if trimmed.is_empty() { "store".to_string() } else { trimmed }
}
