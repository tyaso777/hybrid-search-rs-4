use chrono::TimeZone;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::{mpsc::{self, Receiver, TryRecvError}, Arc};
use std::time::Instant;

fn humanize_bytes(v: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    let nf = v as f64;
    if nf < KB { format!("{} B", v) }
    else if nf < MB { format!("{:.1} KB", nf/KB) }
    else if nf < GB { format!("{:.1} MB", nf/MB) }
    else { format!("{:.1} GB", nf/GB) }
}
use eframe::egui::{self, Button, CentralPanel, ComboBox, ScrollArea, Spinner, TextEdit, DragValue};
use eframe::egui::ProgressBar;
use egui_extras::{Column, TableBuilder, StripBuilder, Size};
use eframe::{App, CreationContext, Frame, NativeOptions};
use rfd::FileDialog;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

use hybrid_service::{HybridService, ServiceConfig, CancelToken, ProgressEvent, HnswState};
use embedding_provider::config::ONNX_STDIO_DEFAULTS;
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
    Files,
    Text,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchMode {
    Hybrid,
    Tantivy,
    Vec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IngestSortKey {
    Default,
    File,
    Size,
    Date,
    Preview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FilesSortKey {
    Default,
    File,
    Size,
    Pages,
    Chunks,
    Updated,
    Author,
    Inserted,
}

#[derive(Debug, Clone)]
enum UiProgressEvent {
    FileStart { index: usize, total: usize, path: String },
    Service(ProgressEvent),
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
    file_path: String,
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
    // Insert Files (folder scan)
    ingest_folder_path: String,
    ingest_exts: String,
    ingest_depth: usize,
    ingest_files: Vec<IngestFileItem>,
    ingest_only_unregistered: bool,
    // Insert Files UI: show absolute paths instead of relative to chosen folder
    ingest_show_abs_paths: bool,
    // Insert Files UI: sorting state
    ingest_sort_key: IngestSortKey,
    ingest_sort_asc: bool,

    // Chunk params (unified for PDF/TXT)
    chunk_min: String,
    chunk_max: String,
    chunk_cap: String,
    chunk_merge_min: String,
    chunk_penalize_short_line: bool,
    chunk_penalize_page_no_nl: bool,

    // Ingest job (async)
    ingest_rx: Option<Receiver<UiProgressEvent>>,
    // For tri‑level progress: per‑file index/total and name
    ingest_file_idx: usize,
    ingest_file_total: usize,
    ingest_file_name: String,
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

    // Prompt builder (Search tab)
    prompt_header_tmpl: String,
    prompt_item_tmpl: String,
    prompt_footer_tmpl: String,
    prompt_items_count: usize,
    prompt_prev: usize,
    prompt_next: usize,
    // Multiple templates support
    prompt_templates: Vec<PromptTemplate>,
    selected_prompt: Option<String>,
    prompt_name_edit: String,
    prompt_name_edit_mode: bool,
    prompt_popup_visible: bool,
    prompt_rendered: String,

    // UI
    tab: ActiveTab,
    insert_mode: InsertMode,
    status: String,
    selected_cid: Option<String>,
    selected_text: String,
    selected_display: String,
    selected_source_path: Option<String>,
    selected_base_cid: Option<String>,
    selected_base_text: String,
    selected_base_display: String,
    selected_base_source_path: Option<String>,
    // Context window for detail view (progressive expand)
    context_chunks: Vec<ContextChunk>,
    context_expanded: bool,
    // Dangerous actions confirmation
    delete_confirm: String,

    // Preview Chunks popup
    preview_visible: bool,
    preview_chunks: Vec<ChunkRecord>,
    preview_selected: Option<usize>,
    preview_show_tab_escape: bool,

    // ONNX Runtime DLL lock (set after first successful Init)
    ort_runtime_committed: Option<String>,
    // Last applied embedder config snapshot (to decide whether to re-init or just apply store paths)
    last_model_path_applied: Option<String>,
    last_tokenizer_path_applied: Option<String>,
    last_embed_dim_applied: Option<usize>,

    // Suggested filename for config save dialog
    config_last_name: String,
    // Optional store name to include in suggested config filename
    config_store_name: String,

    // Track last applied Store Root to auto-apply on Search/Insert
    last_store_root_applied: Option<String>,

    // Validation message for Store Root (shown under the field on failure)
    store_root_error: String,
    // When true, the current service/index may not reflect the UI store yet
    store_paths_stale: bool,

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
    // Files tab: multi-select by DocId
    files_selected_set: std::collections::HashSet<String>,
    // Files tab: sorting
    files_sort_key: FilesSortKey,
    files_sort_asc: bool,
    // Preserve default order across toggles
    files_default_ord: std::collections::HashMap<String, usize>,
}

#[derive(Debug, Clone)]
struct IngestFileItem {
    include: bool,
    path: String,
    size: u64,
    // Original insertion order within a scan
    ordinal: usize,
    // Optional per-file encoding override; None means use global setting
    encoding: Option<String>,
    // Cached preview state for quick mojibake check in the table
    preview_cached_enc: Option<String>,
    preview_cached_text: Option<String>,
    // File modified date in yyyy/mm/dd for display
    modified_ymd: Option<String>,
}

#[derive(Debug, Clone)]
struct ContextChunk {
    cid: String,
    text: String,
    is_base: bool,
}

// New (nested) config format: { store: {...}, chunk: {...}, model: {...}, prompt?: {...} }
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HybridGuiConfigV2 {
    store: StoreCfg,
    chunk: ChunkCfg,
    model: ModelCfg,
    #[serde(default)]
    prompt: Option<PromptCfg>,
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
    #[serde(default)]
    short_merge_min: Option<usize>,
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PromptCfg {
    #[serde(default)]
    templates: Vec<PromptTemplate>,
    #[serde(default)]
    selected: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PromptTemplate {
    name: String,
    header: String,
    item: String,
    footer: String,
    items: usize,
    prev: usize,
    next: usize,
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
    fn navigate_back_to_base(&mut self) {
        if let Some(cid) = self.selected_base_cid.clone() {
            // Restore selection from stored base fields
            self.selected_cid = Some(cid);
            self.selected_text = self.selected_base_text.clone();
            self.selected_display = self.selected_base_display.clone();
            self.selected_source_path = self.selected_base_source_path.clone();
            // Reset context to default (prev/base/next)
            self.rebuild_context_window_initial();
        }
    }

    

    // Initial 3-chunk context: prev/base/next around current base
    fn rebuild_context_window_initial(&mut self) {
        self.context_chunks.clear();
        if !self.ensure_store_paths_current() { return; }
        let Some(base_cid) = self.selected_base_cid.clone() else { return; };
        let base_text = self.selected_base_text.clone();
        match self.fetch_neighbor_chunks(&base_cid) {
            Ok((prev, next)) => {
                if let Some(p) = prev { self.context_chunks.push(ContextChunk { cid: p.chunk_id.0.clone(), text: p.text.clone(), is_base: false }); }
                self.context_chunks.push(ContextChunk { cid: base_cid.clone(), text: base_text, is_base: true });
                if let Some(n) = next { self.context_chunks.push(ContextChunk { cid: n.chunk_id.0.clone(), text: n.text.clone(), is_base: false }); }
            }
            Err(e) => { self.status = format!("Neighbor fetch failed: {e}"); }
        }
        self.context_expanded = false;
    }

    // Expand upward by one chunk (prepend)
    fn expand_context_prev(&mut self) {
        if !self.ensure_store_paths_current() { return; }
        let anchor = if let Some(first) = self.context_chunks.first() { first.cid.clone() } else if let Some(b) = &self.selected_base_cid { b.clone() } else { return; };
        match self.fetch_neighbor_chunks(&anchor) {
            Ok((prev, _)) => {
                if let Some(p) = prev {
                    self.context_chunks.insert(0, ContextChunk { cid: p.chunk_id.0.clone(), text: p.text.clone(), is_base: false });
                    self.context_expanded = true;
                } else { self.status = "No previous chunk".into(); }
            }
            Err(e) => { self.status = format!("Neighbor fetch failed: {e}"); }
        }
    }

    // Expand downward by one chunk (append)
    fn expand_context_next(&mut self) {
        if !self.ensure_store_paths_current() { return; }
        let anchor = if let Some(last) = self.context_chunks.last() { last.cid.clone() } else if let Some(b) = &self.selected_base_cid { b.clone() } else { return; };
        match self.fetch_neighbor_chunks(&anchor) {
            Ok((_, next)) => {
                if let Some(n) = next {
                    self.context_chunks.push(ContextChunk { cid: n.chunk_id.0.clone(), text: n.text.clone(), is_base: false });
                    self.context_expanded = true;
                } else { self.status = "No next chunk".into(); }
            }
            Err(e) => { self.status = format!("Neighbor fetch failed: {e}"); }
        }
    }

    // Get (prev, next) for a chunk id using service/repo
    fn fetch_neighbor_chunks(&self, cid: &str) -> Result<(Option<ChunkRecord>, Option<ChunkRecord>), String> {
        if let Some(svc) = &self.svc {
            svc.neighbor_chunks(cid).map_err(|e| e.to_string())
        } else {
            let repo = chunking_store::sqlite_repo::SqliteRepo::open(self.db_path.trim()).map_err(|e| e.to_string())?;
            repo.get_neighbor_chunks(&ChunkId(cid.to_string())).map_err(|e| e.to_string())
        }
    }
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
            // Bulk delete selected
            let sel_count = self.files_selected_set.len();
            if ui.add_enabled(sel_count > 0 && !self.files_deleting, Button::new(egui::RichText::new(format!("Delete Selected ({})", sel_count)).color(egui::Color32::RED))).clicked() {
                self.delete_selected_files();
            }
        });

        ui.separator();
        // Table
        ui.push_id("files_table", |ui| {
            egui::ScrollArea::horizontal().id_source("files_table_h").show(ui, |ui| {
            let table = TableBuilder::new(ui)
                .striped(true)
                .resizable(true)
                .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                .column(Column::initial(28.0))    // select
                .column(Column::initial(360.0))   // file (source uri)
                .column(Column::initial(72.0))    // size (0.8x)
                .column(Column::initial(56.0))    // pages (0.8x)
                .column(Column::initial(56.0))    // chunks (0.8x)
                .column(Column::initial(136.0))   // updated at (0.8x)
                .column(Column::initial(180.0))   // author
                .column(Column::initial(170.0));  // inserted at

            table
                .header(20.0, |mut header| {
                    // Master checkbox for current page
                    header.col(|ui| {
                        let total = self.files.len();
                        let selected_on_page = self.files.iter().filter(|r| self.files_selected_set.contains(&r.doc_id.0)).count();
                        let mut master = total > 0 && selected_on_page == total;
                        let resp = ui.add(egui::Checkbox::new(&mut master, ""));
                        if resp.clicked() {
                            if master { for rec in &self.files { self.files_selected_set.insert(rec.doc_id.0.clone()); } }
                            else { for rec in &self.files { self.files_selected_set.remove(&rec.doc_id.0); } }
                        }
                        if selected_on_page > 0 && selected_on_page < total { ui.small(format!("{} / {}", selected_on_page, total)); }
                    });
                    header.col(|ui| {
                        let active = matches!(self.files_sort_key, FilesSortKey::File);
                        let arrow = if active { if self.files_sort_asc { " ▲" } else { " ▼" } } else { "" };
                        if ui.button(format!("File{}", arrow)).clicked() {
                            if self.files_sort_key != FilesSortKey::File { self.files_sort_key = FilesSortKey::File; self.files_sort_asc = true; }
                            else if self.files_sort_asc { self.files_sort_asc = false; }
                            else { self.files_sort_key = FilesSortKey::Default; self.files_sort_asc = true; }
                            self.apply_files_sort();
                        }
                    });
                    header.col(|ui| {
                        let active = matches!(self.files_sort_key, FilesSortKey::Size);
                        let arrow = if active { if self.files_sort_asc { " ▲" } else { " ▼" } } else { "" };
                        if ui.button(format!("Size{}", arrow)).clicked() {
                            if self.files_sort_key != FilesSortKey::Size { self.files_sort_key = FilesSortKey::Size; self.files_sort_asc = true; }
                            else if self.files_sort_asc { self.files_sort_asc = false; }
                            else { self.files_sort_key = FilesSortKey::Default; self.files_sort_asc = true; }
                            self.apply_files_sort();
                        }
                    });
                    header.col(|ui| {
                        let active = matches!(self.files_sort_key, FilesSortKey::Pages);
                        let arrow = if active { if self.files_sort_asc { " ▲" } else { " ▼" } } else { "" };
                        if ui.button(format!("Pages{}", arrow)).clicked() {
                            if self.files_sort_key != FilesSortKey::Pages { self.files_sort_key = FilesSortKey::Pages; self.files_sort_asc = true; }
                            else if self.files_sort_asc { self.files_sort_asc = false; }
                            else { self.files_sort_key = FilesSortKey::Default; self.files_sort_asc = true; }
                            self.apply_files_sort();
                        }
                    });
                    header.col(|ui| {
                        let active = matches!(self.files_sort_key, FilesSortKey::Chunks);
                        let arrow = if active { if self.files_sort_asc { " ▲" } else { " ▼" } } else { "" };
                        if ui.button(format!("Chunks{}", arrow)).clicked() {
                            if self.files_sort_key != FilesSortKey::Chunks { self.files_sort_key = FilesSortKey::Chunks; self.files_sort_asc = true; }
                            else if self.files_sort_asc { self.files_sort_asc = false; }
                            else { self.files_sort_key = FilesSortKey::Default; self.files_sort_asc = true; }
                            self.apply_files_sort();
                        }
                    });
                    header.col(|ui| {
                        let active = matches!(self.files_sort_key, FilesSortKey::Updated);
                        let arrow = if active { if self.files_sort_asc { " ▲" } else { " ▼" } } else { "" };
                        if ui.button(format!("Updated{}", arrow)).clicked() {
                            if self.files_sort_key != FilesSortKey::Updated { self.files_sort_key = FilesSortKey::Updated; self.files_sort_asc = true; }
                            else if self.files_sort_asc { self.files_sort_asc = false; }
                            else { self.files_sort_key = FilesSortKey::Default; self.files_sort_asc = true; }
                            self.apply_files_sort();
                        }
                    });
                    header.col(|ui| {
                        let active = matches!(self.files_sort_key, FilesSortKey::Author);
                        let arrow = if active { if self.files_sort_asc { " ▲" } else { " ▼" } } else { "" };
                        if ui.button(format!("Author{}", arrow)).clicked() {
                            if self.files_sort_key != FilesSortKey::Author { self.files_sort_key = FilesSortKey::Author; self.files_sort_asc = true; }
                            else if self.files_sort_asc { self.files_sort_asc = false; }
                            else { self.files_sort_key = FilesSortKey::Default; self.files_sort_asc = true; }
                            self.apply_files_sort();
                        }
                    });
                    header.col(|ui| {
                        let active = matches!(self.files_sort_key, FilesSortKey::Inserted);
                        let arrow = if active { if self.files_sort_asc { " ▲" } else { " ▼" } } else { "" };
                        if ui.button(format!("Inserted{}", arrow)).clicked() {
                            if self.files_sort_key != FilesSortKey::Inserted { self.files_sort_key = FilesSortKey::Inserted; self.files_sort_asc = true; }
                            else if self.files_sort_asc { self.files_sort_asc = false; }
                            else { self.files_sort_key = FilesSortKey::Default; self.files_sort_asc = true; }
                            self.apply_files_sort();
                        }
                    });
                })
                .body(|mut body| {
                    fn humanize_bytes_opt(v: Option<u64>) -> String {
                        match v {
                            Some(n) => {
                                const KB: f64 = 1024.0;
                                const MB: f64 = 1024.0 * 1024.0;
                                const GB: f64 = 1024.0 * 1024.0 * 1024.0;
                                let nf = n as f64;
                                if nf < KB { format!("{} B", n) }
                                else if nf < MB { format!("{:.1} KB", nf/KB) }
                                else if nf < GB { format!("{:.1} MB", nf/MB) }
                                else { format!("{:.1} GB", nf/GB) }
                            }
                            None => String::from("-"),
                        }
                    }
                    // removed unused helper trunc(s, n)
                    for rec in &self.files {
                        body.row(22.0, |mut row_ui| {
                            // select
                            row_ui.col(|ui| {
                                let mut checked = self.files_selected_set.contains(&rec.doc_id.0);
                                if ui.checkbox(&mut checked, "").changed() {
                                    if checked { self.files_selected_set.insert(rec.doc_id.0.clone()); } else { self.files_selected_set.remove(&rec.doc_id.0); }
                                }
                            });
                            row_ui.col(|ui| {
                                let label = egui::Label::new(egui::RichText::new(&rec.source_uri).monospace()).truncate(true).sense(egui::Sense::click());
                                if ui.add(label).clicked() {
                                    self.files_selected_doc = Some(rec.doc_id.0.clone());
                                    self.files_selected_display = rec.source_uri.clone();
                                    self.files_selected_detail = serde_json::to_string_pretty(rec).unwrap_or_else(|_| "<render error>".into());
                                }
                            });
                            row_ui.col(|ui| { ui.label(humanize_bytes_opt(rec.file_size_bytes)); });
                            row_ui.col(|ui| { ui.label(rec.page_count.map(|v| v.to_string()).unwrap_or_else(|| "-".into())); });
                            row_ui.col(|ui| { ui.label(rec.chunk_count.map(|v| v.to_string()).unwrap_or_else(|| "-".into())); });
                            row_ui.col(|ui| { let rawu = rec.updated_at_meta.clone().unwrap_or_else(|| String::from("-")); let disp = format_ts_local_short(&rawu); ui.label(disp); });
                            row_ui.col(|ui| { ui.label(rec.author_guess.clone().unwrap_or_else(|| String::from(""))); });
                            row_ui.col(|ui| { let disp = format_ts_local_short(&rec.extracted_at); ui.label(disp); });
                        });
                    }
                });
            });
        });

        // Per-row delete removed; bulk delete via toolbar

        if let Some(_doc) = &self.files_selected_doc {
            ui.separator();
            if self.files_selected_display.is_empty() {
                ui.label("Selected:");
            } else {
                ui.label(format!("Selected: {}", self.files_selected_display));
            }
            // Open file/folder actions appear before the text content
            if let Some(doc_id) = &self.files_selected_doc {
                if let Some(rec) = self.files.iter().find(|r| &r.doc_id.0 == doc_id) {
                    let path = &rec.source_uri;
                    let (is_local, disp) = normalize_local_path_display(path);
                    ui.horizontal(|ui| {
                        let btn_open = ui.add_enabled(is_local, Button::new("Open file"));
                        if btn_open.clicked() && is_local {
                            if let Some(p) = normalize_local_path(path) { let _ = open_in_os(&p); }
                        }
                        let btn_folder = ui.add_enabled(is_local, Button::new("Open folder"));
                        if btn_folder.clicked() && is_local {
                            if let Some(p) = normalize_local_path(path) { let _ = open_in_os_folder(&p); }
                        }
                        if is_local { ui.monospace(disp); }
                    });
                }
            }
            ScrollArea::vertical().max_height(220.0).id_source("files_selected_scroll").show(ui, |ui| {
                ui.add(TextEdit::multiline(&mut self.files_selected_detail).desired_rows(8).desired_width(800.0).id_source("files_selected_detail"));
            });
        }
    }

    // removed unused method delete_by_doc_id (replaced by bulk delete flow)

    fn delete_selected_files(&mut self) {
        if self.svc.is_none() { self.status = "Service not initialized".into(); return; }
        if !self.ensure_store_paths_current() { return; }
        let ids: Vec<String> = self.files_selected_set.iter().cloned().collect();
        if ids.is_empty() { self.status = "No files selected".into(); return; }
        self.files_deleting = true;
        let mut total_deleted = 0usize;
        if let Some(svc) = &self.svc {
            for doc in &ids {
                let filters = vec![FilterClause { kind: FilterKind::Must, op: FilterOp::DocIdEq(doc.clone()) }];
                match svc.delete_by_filter(&filters, 1000) {
                    Ok(rep) => { total_deleted += rep.db_deleted as usize; },
                    Err(e) => { self.status = format!("Delete failed for {}: {}", doc, e); }
                }
            }
        }
        self.files_selected_set.clear();
        self.files_delete_pending = None;
        self.files_selected_doc = None;
        self.files_selected_display.clear();
        self.files_selected_detail.clear();
        self.files_deleting = false;
        self.refresh_files();
        self.status = format!("Deleted {} records", total_deleted);
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
                Ok(list) => {
                    self.files = list;
                    // Capture default order by doc_id for tri-state sort (Default)
                    self.files_default_ord.clear();
                    for (i, rec) in self.files.iter().enumerate() {
                        self.files_default_ord.insert(rec.doc_id.0.clone(), i);
                    }
                    // Apply current sort selection
                    self.apply_files_sort();
                    self.status = format!("Loaded files page {} ({} items)", self.files_page + 1, self.files.len());
                }
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
        self.store_paths_stale = false;
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
                ui.label("merge<="); ui.add(TextEdit::singleline(&mut self.chunk_merge_min).desired_width(60.0));
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
                short_merge_min: Some(self.chunk_merge_min.trim().parse().unwrap_or(100)),
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
            prompt: Some(PromptCfg {
                templates: self.prompt_templates.clone(),
                selected: self.selected_prompt.clone(),
            }),
        }
    }

    fn apply_ui_config_v2(&mut self, cfg: HybridGuiConfigV2) {
        // Store
        self.store_root = cfg.store.store_root;
        self.refresh_store_paths();
        // Mark paths as stale until user applies or re-inits, but consider them applied for skipping re-apply
        self.store_paths_stale = true;
        self.last_store_root_applied = Some(self.store_root.clone());
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
        self.chunk_merge_min = cfg.chunk.short_merge_min.unwrap_or(100).to_string();
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
        // Prompt templates
        if let Some(p) = cfg.prompt {
            self.prompt_templates = p.templates;
            self.selected_prompt = p.selected;
            // If selected exists, apply to editors
            if let Some(sel) = self.selected_prompt.clone() {
                self.apply_prompt_template_by_name(&sel);
            }
        } else {
            // When not present, keep current UI fields and seed a default
            self.seed_default_prompt_templates_if_empty();
        }
    }

    fn apply_ui_config_v1(&mut self, cfg: HybridGuiConfigV1) {
        // Store
        self.store_root = cfg.store_root;
        self.refresh_store_paths();
        // Mark paths as stale until user applies or re-inits, but consider them applied for skipping re-apply
        self.store_paths_stale = true;
        self.last_store_root_applied = Some(self.store_root.clone());
        // Chunking params
        self.chunk_min = cfg.chunk_min.to_string();
        self.chunk_max = cfg.chunk_max.to_string();
        self.chunk_cap = cfg.chunk_cap.to_string();
        self.chunk_penalize_short_line = cfg.chunk_penalize_short_line;
        self.chunk_penalize_page_no_nl = cfg.chunk_penalize_page_no_nl;
        self.chunk_merge_min = "100".into();
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
            let p = std::path::Path::new(&path);
            if p.exists() {
                // Merge current Prompt Templates into existing config JSON (append/update by name)
                match self.merge_templates_into_config_file(p) {
                    Ok((added, updated)) => {
                        self.status = format!("Updated templates in {} (added {}, updated {})", p.display(), added, updated);
                        if let Some(name) = p.file_name().and_then(|s| s.to_str()) { self.config_last_name = name.to_string(); }
                    }
                    Err(e) => { self.status = format!("Update templates failed: {}", e); }
                }
            } else {
                // Write full config when creating a new file
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
    }

    // Append/merge current prompt templates into an existing config JSON file, preserving other keys.
    fn merge_templates_into_config_file(&self, path: &std::path::Path) -> Result<(usize, usize), String> {
        let s = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut v: serde_json::Value = serde_json::from_str(&s).map_err(|e| e.to_string())?;

        // Ensure prompt object
        if !v.get("prompt").map(|x| x.is_object()).unwrap_or(false) {
            v["prompt"] = serde_json::json!({});
        }
        // Ensure templates array
        if !v["prompt"].get("templates").map(|x| x.is_array()).unwrap_or(false) {
            v["prompt"]["templates"] = serde_json::json!([]);
        }
        let arr = v["prompt"]["templates"].as_array_mut().ok_or("templates is not array")?;

        // Build a map name -> index for existing
        use std::collections::HashMap;
        let mut index: HashMap<String, usize> = HashMap::new();
        for (i, el) in arr.iter().enumerate() {
            if let Some(obj) = el.as_object() {
                if let Some(name) = obj.get("name").and_then(|x| x.as_str()) {
                    index.insert(name.to_string(), i);
                }
            }
        }

        let mut added = 0usize;
        let mut updated = 0usize;
        for tpl in &self.prompt_templates {
            let val = match serde_json::to_value(tpl) { Ok(v) => v, Err(_) => continue };
            if let Some(pos) = index.get(&tpl.name).cloned() {
                if pos < arr.len() { arr[pos] = val; updated += 1; }
            } else {
                arr.push(val);
                index.insert(tpl.name.clone(), arr.len() - 1);
                added += 1;
            }
        }
        // Update selected name
        v["prompt"]["selected"] = match &self.selected_prompt {
            Some(n) => serde_json::Value::String(n.clone()),
            None => serde_json::Value::Null,
        };

        let body = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
        std::fs::write(path, body).map_err(|e| e.to_string())?;
        Ok((added, updated))
    }

    fn import_prompt_templates_via_dialog(&mut self) {
        if let Some(path) = FileDialog::new().add_filter("JSON", &["json"]).pick_file() {
            match std::fs::read_to_string(&path) {
                Ok(s) => {
                    match serde_json::from_str::<HybridGuiConfigV2>(&s) {
                        Ok(cfg2) => {
                            if let Some(p) = cfg2.prompt {
                                let mut added = 0usize;
                                let mut updated = 0usize;
                                for tpl in p.templates {
                                    if let Some(pos) = self.prompt_templates.iter().position(|t| t.name == tpl.name) {
                                        self.prompt_templates[pos] = tpl;
                                        updated += 1;
                                    } else {
                                        self.prompt_templates.push(tpl);
                                        added += 1;
                                    }
                                }
                                self.status = format!("Imported templates from {} (added {}, updated {})", path.display(), added, updated);
                            } else {
                                self.status = format!("No prompt templates found in {}", path.display());
                            }
                        }
                        Err(_) => {
                            self.status = format!("Import failed: not a v2 config with prompts: {}", path.display());
                        }
                    }
                }
                Err(e) => { self.status = format!("Read failed: {}", e); }
            }
        }
    }

    fn save_config_overwrite_via_dialog(&mut self) {
        let suggested = self.suggest_config_filename();
        if let Some(path) = FileDialog::new()
            .add_filter("JSON", &["json"]).set_file_name(&suggested)
            .save_file()
        {
            let cfg = self.to_ui_config_v2();
            match serde_json::to_string_pretty(&cfg) {
                Ok(body) => match std::fs::write(&path, body) {
                    Ok(_) => {
                        self.status = format!("Saved config (overwrite) to {}", path.display());
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
        let base = "hybrid_service_config";
        let name = self.config_store_name.trim();
        if name.is_empty() {
            format!("{}.json", base)
        } else {
            let safe = safe_filename_component(name);
            format!("{}_{}.json", base, safe)
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
        let store_default = String::from("target/demo/store");
        let mut s = Self {
            model_path: String::from("embedding_provider/models/ruri-v3-onnx/model.onnx"),
            tokenizer_path: String::from("embedding_provider/models/ruri-v3-onnx/tokenizer.json"),
            runtime_path: String::from("embedding_provider/bin/onnxruntime-win-x64-1.23.1/lib/onnxruntime.dll"),
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
            ingest_folder_path: String::new(),
            ingest_exts: String::from("pdf, docx, pptx, xlsx, xls, ods, txt, md, markdown, csv, tsv, log, json, yaml, yml, ini, toml, cfg, conf, rst, tex, srt, properties"),
            ingest_depth: 1,
            ingest_files: Vec::new(),
            ingest_only_unregistered: true,
            ingest_show_abs_paths: false,
            ingest_sort_key: IngestSortKey::Default,
            ingest_sort_asc: true,

            chunk_min: String::from("400"),
            chunk_max: String::from("600"),
            chunk_cap: String::from("800"),
            chunk_merge_min: String::from("100"),
            chunk_penalize_short_line: true,
            chunk_penalize_page_no_nl: true,

            ingest_rx: None,
            ingest_file_idx: 0,
            ingest_file_total: 0,
            ingest_file_name: String::new(),
            ingest_cancel: None,
            ingest_running: false,
            ingest_done: 0,
            ingest_total: 0,
            ingest_last_batch: 0,
            ingest_started: None,
            ingest_doc_key: None,

            query: String::new(),
            top_k: 10,
            // Default weights: favor VEC over TV (1:4). TV/AND left unset.
            w_tv: None,
            w_tv_and: None,
            w_tv_or: Some(1.0),
            w_vec: Some(4.0),
            results: Vec::new(),
            search_mode: SearchMode::Hybrid,

            // Prompt defaults (JSON style)
            // Instruction: fixed directive; query is filled separately
            prompt_header_tmpl: String::from(
                "{\n  \"instruction\": \"Use the provided results from vector/BM25 search to answer the user’s query. Cite the rank numbers of all items you relied on (e.g., #2, #5). If the results are insufficient to answer, say so and avoid speculation. Answer in Japanese.\",\n  \"query\": \"<<Query:escape_json>>\",\n  \"results\": [\n"
            ),
            prompt_item_tmpl: String::from("{\"rank\": <<Rank>>, \"file\": \"<<File:escape_json>>\", \"page\": \"<<Page:escape_json>>\", \"text\": \"<<Text:escape_json>>\"}<<Comma>>"),
            prompt_footer_tmpl: String::from("  ]\n}\n"),
            prompt_items_count: 5,
            prompt_prev: 1,
            prompt_next: 1,
            prompt_templates: Vec::new(),
            selected_prompt: None,
            prompt_name_edit: String::new(),
            prompt_name_edit_mode: false,
            prompt_popup_visible: false,
            prompt_rendered: String::new(),

            tab: ActiveTab::Config,
            insert_mode: InsertMode::File,
            status: String::new(),
            selected_cid: None,
            selected_text: String::new(),
            selected_display: String::new(),
            selected_source_path: None,
            selected_base_cid: None,
            selected_base_text: String::new(),
            selected_base_display: String::new(),
            selected_base_source_path: None,
            context_chunks: Vec::new(),
            context_expanded: false,
            delete_confirm: String::new(),

            preview_visible: false,
            preview_chunks: Vec::new(),
            preview_selected: None,
            preview_show_tab_escape: true,

            ort_runtime_committed: None,
            last_model_path_applied: None,
            last_tokenizer_path_applied: None,
            last_embed_dim_applied: None,

            config_last_name: String::from("hybrid_service_config.json"),
            config_store_name: String::new(),

            last_store_root_applied: None,

            store_root_error: String::new(),
            store_paths_stale: false,

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
            files_selected_set: std::collections::HashSet::new(),
            files_sort_key: FilesSortKey::Default,
            files_sort_asc: true,
            files_default_ord: std::collections::HashMap::new(),
        };
        // Ensure the default prompt header uses the directive above for `instruction`
        s.prompt_header_tmpl = String::from(
            "{\n  \"instruction\": \"Use the provided results from vector/BM25 search to answer the user’s query. Cite the rank numbers of all items you relied on (e.g., #2, #5). If the results are insufficient to answer, say so and avoid speculation. Answer in Japanese.\",\n  \"query\": \"<<Query:escape_json>>\",\n  \"results\": [\n"
        );
        // Seed default prompt template list
        s.seed_default_prompt_templates_if_empty();
        s
    }

    fn seed_default_prompt_templates_if_empty(&mut self) {
        if self.prompt_templates.is_empty() {
            let name = "JSON Default".to_string();
            self.prompt_templates.push(PromptTemplate {
                name: name.clone(),
                header: self.prompt_header_tmpl.clone(),
                item: self.prompt_item_tmpl.clone(),
                footer: self.prompt_footer_tmpl.clone(),
                items: self.prompt_items_count,
                prev: self.prompt_prev,
                next: self.prompt_next,
            });
            self.selected_prompt = Some(name);
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
        // Decide: re-init embedder or just apply store paths
        let dim_ui: usize = self.embedding_dimension.trim().parse().unwrap_or(ONNX_STDIO_DEFAULTS.embedding_dimension);
        let embedder_changed = match &self.svc {
            None => true,
            Some(_) => {
                let same_model = self.last_model_path_applied.as_deref() == Some(self.model_path.trim());
                let same_tok = self.last_tokenizer_path_applied.as_deref() == Some(self.tokenizer_path.trim());
                let same_rt = self.ort_runtime_committed.as_deref() == Some(self.runtime_path.trim());
                let same_dim = self.last_embed_dim_applied == Some(dim_ui);
                !(same_model && same_tok && same_rt && same_dim)
            }
        };

        if !embedder_changed {
            // Fast path: apply only store paths on existing service
            if let Some(svc) = &self.svc {
                svc.set_store_paths(PathBuf::from(db.clone()), Some(PathBuf::from(hnsw.clone())));
                self.status = "Applied store paths (HNSW reload in background)".into();
                self.store_paths_stale = false;
                self.last_store_root_applied = Some(root.clone());
            } else {
                // Shouldn't happen, but fall back to full init
                let (tx, rx) = mpsc::channel();
                self.status = "Initializing model...".into();
                self.store_paths_stale = false;
                self.svc_task = Some(ServiceInitTask { rx, started: Instant::now() });
                std::thread::spawn(move || {
                    let mut cfg = ServiceConfig::default();
                    cfg.db_path = PathBuf::from(db);
                    cfg.hnsw_dir = Some(PathBuf::from(hnsw));
                    let _ = tx.send(HybridService::new(cfg).map(|s| Arc::new(s)).map_err(|e| e.to_string()));
                });
            }
            return;
        }

        // Full re-init path
        let mut cfg = ServiceConfig::default();
        cfg.db_path = PathBuf::from(db);
        cfg.hnsw_dir = Some(PathBuf::from(hnsw));
        cfg.embedder.model_path = PathBuf::from(self.model_path.trim());
        cfg.embedder.tokenizer_path = PathBuf::from(self.tokenizer_path.trim());
        cfg.embedder.runtime_library_path = PathBuf::from(self.runtime_path.trim());
        cfg.embedder.dimension = dim_ui;
        cfg.embedder.max_input_length = self.max_tokens.trim().parse().unwrap_or(ONNX_STDIO_DEFAULTS.max_input_tokens);
        if let Ok(bs) = self.embed_batch_size.trim().parse::<usize>() { if bs > 0 { cfg.embed_batch_size = bs; } }
        cfg.embed_auto = self.embed_auto;
        if let Ok(x) = self.embed_initial_batch.trim().parse::<usize>() { if x > 0 { cfg.embed_initial_batch = x; } }
        if let Ok(x) = self.embed_min_batch.trim().parse::<usize>() { if x > 0 { cfg.embed_min_batch = x; } }

        let (tx, rx) = mpsc::channel();
        self.status = "Initializing model...".into();
        self.store_paths_stale = false;
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
                    // Mark current Store Root as applied and indices fresh
                    self.last_store_root_applied = Some(self.store_root.trim().to_string());
                    self.store_paths_stale = false;
                    // Record last applied embedder config snapshot
                    self.ort_runtime_committed = Some(self.runtime_path.trim().to_string());
                    self.last_model_path_applied = Some(self.model_path.trim().to_string());
                    self.last_tokenizer_path_applied = Some(self.tokenizer_path.trim().to_string());
                    self.last_embed_dim_applied = Some(self.embedding_dimension.trim().parse().unwrap_or(ONNX_STDIO_DEFAULTS.embedding_dimension));
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
                    // Top-level tabs with underline accent (Config first)
                    let resp_config = ui.selectable_value(&mut self.tab, ActiveTab::Config, "Config");
                    let resp_insert = ui.selectable_value(&mut self.tab, ActiveTab::Insert, "Insert");
                    let resp_search = ui.selectable_value(&mut self.tab, ActiveTab::Search, "Search");
                    let resp_files = ui.selectable_value(&mut self.tab, ActiveTab::Files, "Files");
                    let union12 = resp_config.rect.union(resp_insert.rect);
                    let union123 = union12.union(resp_search.rect);
                    let union_all = union123.union(resp_files.rect);
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
                    // Color-coded status: warn color for not-ready/editable; green-ish for ready/fixed; red for error
                    let warn = ui.visuals().warn_fg_color;
                    let ok = egui::Color32::from_rgb(58, 166, 94);
                    let err = egui::Color32::RED;
                    let model_color = match model_status { "ready" => ok, "loading" | "released" => warn, _ => err };
                    let dll_color = match dll_status { "fixed" => ok, "editable" => warn, _ => warn };
                    let index_color = match index_status { "ready" => ok, "loading" | "absent" => warn, "error" => err, _ => warn };
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("Model: {}", model_status)).color(model_color));
                ui.label(", ");
                ui.label(egui::RichText::new(format!("RuntimeDLL: {}", dll_status)).color(dll_color));
                ui.label(", ");
                // Index indicator priority: init in progress -> loading, else stale flag, else service-reported state
                let idx_disp = if self.svc_task.is_some() { "loading" }
                    else if self.store_paths_stale { "stale" }
                    else { index_status };
                let idx_color = if self.svc_task.is_some() || self.store_paths_stale { ui.visuals().warn_fg_color } else { index_color };
                ui.label(egui::RichText::new(format!("Index: {}", idx_disp)).color(idx_color));
            });
                    // compact controls with a clear right-side divider as well
                    ui.add_space(8.0);
                    ui.add_space(8.0);
                    if ui.button("Init").clicked() {
                        // Always run init/apply flow: model re-init when embedder changed, otherwise store apply only
                        self.start_service_init();
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
        // Popups (render after main UI so they appear above)
        self.ui_preview_window(ctx);
        self.ui_prompt_window(ctx);
    }
}

impl AppState {
    fn ui_insert(&mut self, ui: &mut egui::Ui) {
        ui.heading("Insert");
            // Sub-tabs for Insert
            ui.horizontal(|ui| {
                // Tab buttons with underline
                let resp_file = ui.selectable_value(&mut self.insert_mode, InsertMode::File, "Insert File");
                let resp_files = ui.selectable_value(&mut self.insert_mode, InsertMode::Files, "Insert Files");
                let resp_text = ui.selectable_value(&mut self.insert_mode, InsertMode::Text, "Insert Text");

                // Compute a union rect spanning all buttons
                let union12 = resp_file.rect.union(resp_files.rect);
                let union = union12.union(resp_text.rect);
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
                    InsertMode::Files => resp_files.rect,
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
                    ui.add_enabled_ui(!self.ingest_running, |ui| {
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
                        if self.ingest_preview.chars().count() > 60 { preview_short.push('\u{2026}'); }
                        ui.label(format!("Preview: {}", preview_short.replace(['\n','\r','\t'], " ")));
                    }

                    // Row 4: [Preview Chunks] button
                    if ui.add_enabled(!self.ingest_running, Button::new("Preview Chunks")).clicked() {
                        self.do_preview_chunks();
                    }

                    // Row 5: Ingest File
                    if ui.add_enabled(!self.ingest_running, Button::new("Ingest File")).clicked() { self.do_ingest_file(); }
                    });
                },
                InsertMode::Files => {
                    // Disable interactive controls while ingesting
                    ui.add_enabled_ui(!self.ingest_running, |ui| {
                        // Row 1: Choose Folder, [Folder path]
                        ui.horizontal(|ui| {
                            if ui.button("Choose Folder").clicked() {
                                if let Some(p) = FileDialog::new().pick_folder() {
                                    self.ingest_folder_path = p.display().to_string();
                                }
                            }
                            ui.add(TextEdit::singleline(&mut self.ingest_folder_path).desired_width(420.0));
                        });
                        // Row 2: Extensions + Encoding + Depth
                        ui.horizontal(|ui| { ui.label("Extensions (comma)");
                            ui.add(TextEdit::singleline(&mut self.ingest_exts).desired_width(220.0));
                            ui.label("Encoding");
                            let encs = ["auto", "utf-8", "shift_jis", "windows-1252", "utf-16le", "utf-16be"];
                            ComboBox::from_id_source("ingest_files_encoding")
                                .selected_text(self.ingest_encoding.clone())
                                .show_ui(ui, |ui| {
                                    for e in encs { ui.selectable_value(&mut self.ingest_encoding, e.to_string(), e); }
                                });
                            ui.label("Depth");
                            ui.add(DragValue::new(&mut self.ingest_depth).clamp_range(0..=16));
                        });
                        // Row 3: Scan buttons under Extensions
                        ui.horizontal(|ui| { ui.checkbox(&mut self.ingest_only_unregistered, "Check unregistered files"); });
                        ui.horizontal(|ui| {
                            if ui.button("Scan").clicked() {
                                if self.ingest_only_unregistered { self.scan_ingest_folder_unregistered(); } else { self.scan_ingest_folder(); }
                            }
                        ui.separator();
                        });
                        ui.horizontal(|ui| {
                            let has_items = !self.ingest_files.is_empty();
                            let selected_count = self.ingest_files.iter().filter(|i| i.include).count();
                            let any_selected = selected_count > 0;
                            let can_ingest = has_items && any_selected && !self.ingest_running;
                            if ui.add_enabled(can_ingest, Button::new("Ingest Selected")).clicked() {
                                self.do_ingest_files_batch();
                            }
                            // Show selection summary
                            if has_items {
                                ui.label(format!("Selected: {} / Total: {}", selected_count, self.ingest_files.len()));
                            }
                            ui.checkbox(&mut self.ingest_show_abs_paths, "Absolute paths");
                        });
                    }); // end disabled controls block
                    // Inline progress under the button (two bars): chunk progress and file progress
                    if self.ingest_running {
                        // 1) Chunk progress
                        let frac_chunks: f32 = if self.ingest_total > 0 { (self.ingest_done as f32 / self.ingest_total as f32).clamp(0.0, 1.0) } else { 0.0 };
                        ui.horizontal(|ui| {
                            ui.add(ProgressBar::new(frac_chunks).desired_width(420.0).show_percentage());
                            ui.label(format!("{} / {} (batch {})", self.ingest_done, self.ingest_total, self.ingest_last_batch));
                        });
                        // 2) File progress (finished files count over total)
                        if self.ingest_file_total > 0 {
                            let finished_files = self.ingest_file_idx.saturating_sub(1) as f32;
                            let total_files = self.ingest_file_total as f32;
                            let frac_files: f32 = (finished_files / total_files).clamp(0.0, 1.0);
                            ui.horizontal(|ui| {
                                ui.add(ProgressBar::new(frac_files).desired_width(420.0).show_percentage());
                                ui.label(format!("File {} / {}: {}", self.ingest_file_idx, self.ingest_file_total, self.ingest_file_name));
                            });
                        }
                    }
                    ui.add_space(4.0);
                    // Results table or empty message (disabled during ingest)
                    ui.add_enabled_ui(!self.ingest_running, |ui| {
                        if self.ingest_files.is_empty() {
                            ui.label("No files scanned.");
                        } else {
                            let tbl_id = if self.ingest_show_abs_paths { "ingest_files_table_abs" } else { "ingest_files_table_rel" };
                            ui.push_id(tbl_id, |ui| {
                                egui::ScrollArea::horizontal().id_source("ingest_files_table_h").show(ui, |ui| {
                                    // Use a narrower default for the File column when showing relative paths
                                    let file_col_w: f32 = if self.ingest_show_abs_paths { 520.0 } else { 520.0 * 0.7 };
                                    let table = TableBuilder::new(ui)
                                        .striped(true)
                                        .resizable(true)
                                        .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                                        .column(Column::initial(28.0))    // include checkbox
                                        .column(Column::initial(file_col_w))   // File (path)
                                        .column(Column::initial(72.0))    // Size (0.6x)
                                        .column(Column::initial(80.0))    // Date (yyyy/mm/dd, 0.8x)
                                        .column(Column::initial(112.0))   // Encoding override (0.8x)
                                        .column(Column::initial(220.0));  // Preview (first chars)

                                    table
                                        .header(20.0, |mut header| {
                                        header.col(|ui| {
                                            let total = self.ingest_files.len();
                                            let selected = self.ingest_files.iter().filter(|i| i.include).count();
                                            let mut master = total > 0 && selected == total;
                                            let resp = ui.add(egui::Checkbox::new(&mut master, ""));
                                            if resp.clicked() {
                                                for it in &mut self.ingest_files { it.include = master; }
                                            }
                                            if selected > 0 && selected < total {
                                                ui.small(format!("{} / {}", selected, total));
                                            }
                                        });
                                        header.col(|ui| {
                                            let active = matches!(self.ingest_sort_key, IngestSortKey::File);
                                            let arrow = if active { if self.ingest_sort_asc { " ▲" } else { " ▼" } } else { "" };
                                            if ui.button(format!("File{}", arrow)).clicked() {
                                                if self.ingest_sort_key != IngestSortKey::File { self.ingest_sort_key = IngestSortKey::File; self.ingest_sort_asc = true; } else if self.ingest_sort_asc { self.ingest_sort_asc = false; } else { self.ingest_sort_key = IngestSortKey::Default; self.ingest_sort_asc = true; }
                                                { let base = self.ingest_folder_path.clone(); let abs = self.ingest_show_abs_paths; self.apply_ingest_sort_with(base.as_str(), abs) };
                                            }
                                        });
                                        header.col(|ui| {
                                            let active = matches!(self.ingest_sort_key, IngestSortKey::Size);
                                            let arrow = if active { if self.ingest_sort_asc { " ▲" } else { " ▼" } } else { "" };
                                            if ui.button(format!("Size{}", arrow)).clicked() {
                                                if self.ingest_sort_key != IngestSortKey::Size { self.ingest_sort_key = IngestSortKey::Size; self.ingest_sort_asc = true; } else if self.ingest_sort_asc { self.ingest_sort_asc = false; } else { self.ingest_sort_key = IngestSortKey::Default; self.ingest_sort_asc = true; }
                                                { let base = self.ingest_folder_path.clone(); let abs = self.ingest_show_abs_paths; self.apply_ingest_sort_with(base.as_str(), abs) };
                                            }
                                        });
                                        header.col(|ui| {
                                            let active = matches!(self.ingest_sort_key, IngestSortKey::Date);
                                            let arrow = if active { if self.ingest_sort_asc { " ▲" } else { " ▼" } } else { "" };
                                            if ui.button(format!("Date{}", arrow)).clicked() {
                                                if self.ingest_sort_key != IngestSortKey::Date { self.ingest_sort_key = IngestSortKey::Date; self.ingest_sort_asc = true; } else if self.ingest_sort_asc { self.ingest_sort_asc = false; } else { self.ingest_sort_key = IngestSortKey::Default; self.ingest_sort_asc = true; }
                                                { let base = self.ingest_folder_path.clone(); let abs = self.ingest_show_abs_paths; self.apply_ingest_sort_with(base.as_str(), abs) };
                                            }
                                        });
                                        header.col(|ui| { ui.label("Encoding"); });
                                        header.col(|ui| {
                                            let active = matches!(self.ingest_sort_key, IngestSortKey::Preview);
                                            let arrow = if active { if self.ingest_sort_asc { " ▲" } else { " ▼" } } else { "" };
                                            if ui.button(format!("Preview (text){}", arrow)).clicked() {
                                                if self.ingest_sort_key != IngestSortKey::Preview { self.ingest_sort_key = IngestSortKey::Preview; self.ingest_sort_asc = true; } else if self.ingest_sort_asc { self.ingest_sort_asc = false; } else { self.ingest_sort_key = IngestSortKey::Default; self.ingest_sort_asc = true; }
                                                { let base = self.ingest_folder_path.clone(); let abs = self.ingest_show_abs_paths; self.apply_ingest_sort_with(base.as_str(), abs) };
                                            }
                                        });
                                    })
                                    .body(|mut body| {
                                        let base_root = self.ingest_folder_path.clone();
                                        let show_abs = self.ingest_show_abs_paths;
                                        for it in &mut self.ingest_files {
                                            body.row(22.0, |mut row| {
                                                row.col(|ui| { ui.checkbox(&mut it.include, ""); });
                                                row.col(|ui| {
                                                    let disp = display_path_with_root(&it.path, base_root.as_str(), show_abs);
                                                    ui.monospace(disp);
                                                });
                                                row.col(|ui| { ui.label(humanize_bytes(it.size)); });
                                                // Date before Encoding
                                                row.col(|ui| { ui.label(it.modified_ymd.clone().unwrap_or_else(|| String::from("-"))); });
                                                row.col(|ui| {
                                                    let encs = ["(global)", "utf-8", "shift_jis", "windows-1252", "utf-16le", "utf-16be"];
                                                    let current = it.encoding.clone().unwrap_or_else(|| String::from("(global)"));
                                                    let mut sel = current.clone();
                                                    ComboBox::from_id_source(format!("enc_{}", it.path))
                                                        .selected_text(sel.clone())
                                                        .show_ui(ui, |ui| {
                                                            for e in encs { ui.selectable_value(&mut sel, e.to_string(), e); }
                                                        });
                                                    if sel == "(global)" { it.encoding = None; } else { it.encoding = Some(sel); }
                                                });
                                                row.col(|ui| {
                                                    // Show a short preview for text-like files with the effective encoding
                                                    let lower = it.path.to_ascii_lowercase();
                                                    let text_exts = [
                                                        ".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".json", ".yaml", ".yml",
                                                        ".ini", ".toml", ".cfg", ".conf", ".rst", ".tex", ".srt", ".properties",
                                                    ];
                                                    let is_text_like = text_exts.iter().any(|e| lower.ends_with(e));
                                                    if is_text_like {
                                                        let enc_eff = it.encoding.clone().unwrap_or_else(|| self.ingest_encoding.clone());
                                                        let need_reload = it.preview_cached_enc.as_deref() != Some(enc_eff.as_str());
                                                        if need_reload || it.preview_cached_text.is_none() {
                                                            let pv = preview_text_for_file(&it.path, &enc_eff, 4096, 48).unwrap_or_else(|| String::from("-"));
                                                            it.preview_cached_enc = Some(enc_eff);
                                                            it.preview_cached_text = Some(pv);
                                                        }
                                                        ui.label(it.preview_cached_text.as_deref().unwrap_or("-"));
                                                    } else {
                                                        ui.label("-");
                                                    }
                                                });
                                            });
                                        }
                                    });
                            });
                        });
                        // (Ingest button moved above the table)
                    }
                    }); // end disabled table block
                },
                InsertMode::Text => {
                    ui.add_enabled_ui(!self.ingest_running, |ui| {
                    // Text input with the action button placed below
                    ui.label("Text");
                    ui.add(TextEdit::multiline(&mut self.input_text).desired_rows(4).desired_width(600.0));
                    if ui.add(Button::new("Insert Text")).clicked() { self.do_insert_text(); }
                    });
                }
            }
        // Preview popup is rendered after main UI in update()
        // Moved progress bars next to the Ingest Selected button; no extra bar here.
    }

    fn do_preview_chunks(&mut self) {
        let path = self.ingest_file_path.trim();
        if path.is_empty() { self.status = "Pick a file to preview".into(); return; }
        // Build params from UI
        let mut params = file_chunker::text_segmenter::TextChunkParams::default();
        if let Ok(v) = self.chunk_min.trim().parse::<usize>() { if v > 0 { params.min_chars = v; } }
        if let Ok(v) = self.chunk_max.trim().parse::<usize>() { if v > 0 { params.max_chars = v; } }
        if let Ok(v) = self.chunk_cap.trim().parse::<usize>() { if v > 0 { params.cap_chars = v; } }
        if let Ok(v) = self.chunk_merge_min.trim().parse::<usize>() { params.short_merge_min_chars = v; }
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
                            egui::ScrollArea::vertical()
                                .id_source("preview_chunks_list")
                                .auto_shrink([false, false])
                                .show(ui, |ui| {
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
                                    egui::ScrollArea::vertical()
                                        .id_source("preview_selected_text")
                                        .show(ui, |ui| { ui.monospace(text); });

                                    // Metadata and JSON detail sections removed for a simpler view
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
        let (tx, rx) = mpsc::channel::<UiProgressEvent>();
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
        let merge_min = self.chunk_merge_min.trim().parse().unwrap_or(100);
        std::thread::spawn(move || {
            let hint = doc_hint_opt.as_deref();
            let tx2 = tx.clone(); let cb: Box<dyn FnMut(ProgressEvent) + Send> = Box::new(move |ev: ProgressEvent| { let _ = tx2.send(UiProgressEvent::Service(ev)); });
            let _ = tx.send(UiProgressEvent::FileStart { index: 1, total: 1, path: path_owned.clone() });
            let _ = svc.ingest_file_with_progress_custom(
                &path_owned, hint,
                enc_opt.as_deref(),
                min, max, cap, merge_min, ps, pp,
                Some(&cancel), Some(cb)
            );
            // The service emits Finished/Canceled; no-op here.
        });
    }

    fn scan_ingest_folder(&mut self) {
        self.ingest_files.clear();
        let root = self.ingest_folder_path.trim();
        if root.is_empty() { self.status = "Choose a folder".into(); return; }
        let max_depth = self.ingest_depth;
        let filters: Vec<String> = self.ingest_exts.split(',').map(|s| s.trim().trim_start_matches('.').to_ascii_lowercase()).filter(|s| !s.is_empty()).collect();
        let use_all = filters.is_empty() || filters.iter().any(|f| f == "*");
        let mut stack: Vec<(std::path::PathBuf, usize)> = vec![(std::path::PathBuf::from(root), 0)];
        let mut seq: usize = 0;
        while let Some((dir, depth)) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for entry in rd.flatten() {
                let path = entry.path();
                if let Ok(meta) = entry.metadata() {
                    if meta.is_dir() {
                        if depth < max_depth { stack.push((path, depth + 1)); }
                    } else if meta.is_file() {
                        let pstr = path.display().to_string();
                        let lower = pstr.to_ascii_lowercase();
                        let matched = if use_all { true } else {
                            let mut ok = false;
                            for f in &filters {
                                if lower.ends_with(&format!(".{}", f)) { ok = true; break; }
                            }
                            ok
                        };
                        if matched {
                            let mdy = meta.modified().ok().map(|st| format_ymd(st));
                            self.ingest_files.push(IngestFileItem { include: true, path: pstr, size: meta.len(), ordinal: seq, encoding: None, preview_cached_enc: None, preview_cached_text: None, modified_ymd: mdy });
                            seq += 1;
                        }
                    }
                }
            }
        }
        { let base = self.ingest_folder_path.clone(); let abs = self.ingest_show_abs_paths; self.apply_ingest_sort_with(base.as_str(), abs) };
        self.status = format!("Scanned {} files", self.ingest_files.len());
    }

    fn scan_ingest_folder_unregistered(&mut self) {
        if !self.ensure_store_paths_current() { return; }
        let Some(svc) = &self.svc else { self.status = "Model not initialized".into(); return; };

        use std::collections::HashSet;
        let mut known: HashSet<String> = HashSet::new();
        let mut known_sizes: HashSet<u64> = HashSet::new();
        let limit: usize = 1000;
        let mut offset: usize = 0;
        loop {
            match svc.list_files(limit, offset) {
                Ok(list) => {
                    if list.is_empty() { break; }
                    for rec in &list {
                        if let Some(h) = rec.content_sha256.clone() { known.insert(h); }
                        if let Some(sz) = rec.file_size_bytes { known_sizes.insert(sz); }
                    }
                    if list.len() < limit { break; }
                    offset += limit;
                }
                Err(e) => { self.status = format!("Fetch known hashes failed: {e}"); return; }
            }
        }

        self.ingest_files.clear();
        let root = self.ingest_folder_path.trim();
        if root.is_empty() { self.status = "Choose a folder".into(); return; }
        let max_depth = self.ingest_depth;
        let filters: Vec<String> = self.ingest_exts
            .split(',')
            .map(|s| s.trim().trim_start_matches('.').to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        let use_all = filters.is_empty() || filters.iter().any(|f| f == "*");
        let mut stack: Vec<(std::path::PathBuf, usize)> = vec![(std::path::PathBuf::from(root), 0)];
        let mut total: usize = 0; let mut kept: usize = 0;
        let mut immediate: Vec<IngestFileItem> = Vec::new();
        let mut needs_hash: Vec<(String, u64, Option<String>, usize)> = Vec::new();
        let mut seq: usize = 0;
        while let Some((dir, depth)) = stack.pop() {
            let Ok(rd) = std::fs::read_dir(&dir) else { continue };
            for entry in rd.flatten() {
                let path = entry.path();
                if let Ok(meta) = entry.metadata() {
                    if meta.is_dir() {
                        if depth < max_depth { stack.push((path, depth + 1)); }
                    } else if meta.is_file() {
                        let pstr = path.display().to_string();
                        let lower = pstr.to_ascii_lowercase();
                        let matched = if use_all { true } else {
                            let mut ok = false;
                            for f in &filters { if lower.ends_with(&format!(".{}", f)) { ok = true; break; } }
                            ok
                        };
                        if !matched { continue; }
                        total += 1;
                        let fsz = meta.len();
                        if !known_sizes.contains(&fsz) {
                            // No DB file with this size: definitely new, no hashing required
                            { let mdy = meta.modified().ok().map(|st| format_ymd(st)); immediate.push(IngestFileItem { include: true, path: pstr, size: fsz, ordinal: seq, encoding: None, preview_cached_enc: None, preview_cached_text: None, modified_ymd: mdy }); seq += 1; }
                            kept += 1;
                        } else {
                            // Size exists in DB: queue for hashing
                            { let mdy = meta.modified().ok().map(|st| format_ymd(st)); needs_hash.push((pstr, fsz, mdy, seq)); seq += 1; }
                        }
                    }
                }
            }
        }

        // Hash the queued files in parallel
        let hashed_results: Vec<IngestFileItem> = needs_hash
            .par_iter()
            .map(|(p, fsz, mdy, ord)| {
                let mut include = true;
                if let Some(hx) = sha256_hex_file(p) {
                    if known.contains(&hx) { include = false; }
                }
                IngestFileItem { include, path: p.clone(), size: *fsz, ordinal: *ord, encoding: None, preview_cached_enc: None, preview_cached_text: None, modified_ymd: mdy.clone() }
            })
            .collect();

        for it in &hashed_results { if it.include { kept += 1; } }

        self.ingest_files.clear();
        self.ingest_files.extend(immediate);
        self.ingest_files.extend(hashed_results);
        { let base = self.ingest_folder_path.clone(); let abs = self.ingest_show_abs_paths; self.apply_ingest_sort_with(base.as_str(), abs) };
        self.status = format!("Scanned {} files (new: {})", total, kept);
    }
    fn do_ingest_files_batch(&mut self) {
        if !self.ensure_store_paths_current() { return; }
        let Some(svc) = self.svc.as_ref() else { self.status = "Model not initialized".into(); return; };
        let selected: Vec<(String, Option<String>)> = self.ingest_files
            .iter()
            .filter(|i| i.include)
            .map(|i| (i.path.clone(), i.encoding.clone()))
            .collect();
        if selected.is_empty() { self.status = "No files selected".into(); return; }
        let svc = Arc::clone(svc);
        let (tx, rx) = mpsc::channel::<UiProgressEvent>();
        let cancel = CancelToken::new();
        self.ingest_rx = Some(rx);
        self.ingest_cancel = Some(cancel.clone());
        self.ingest_running = true;
        self.ingest_done = 0;
        self.ingest_total = 0;
        self.ingest_last_batch = 0;
        self.ingest_started = Some(Instant::now());
        self.status = format!("Ingesting {} files...", selected.len());
        let enc_opt = { let e = self.ingest_encoding.trim().to_string(); if e.is_empty() || e.eq_ignore_ascii_case("auto") { None } else { Some(e) } };
        let min = self.chunk_min.trim().parse().unwrap_or(400);
        let max = self.chunk_max.trim().parse().unwrap_or(600);
        let cap = self.chunk_cap.trim().parse().unwrap_or(800);
        let ps = self.chunk_penalize_short_line;
        let pp = self.chunk_penalize_page_no_nl;
        let merge_min = self.chunk_merge_min.trim().parse().unwrap_or(100);
        let doc_hint = if self.doc_hint.trim().is_empty() { None } else { Some(self.doc_hint.trim().to_string()) };
        std::thread::spawn(move || {
            for (idx, (p, enc_override)) in selected.iter().enumerate() {
                let hint = doc_hint.as_deref();
                let tx2 = tx.clone(); let cb: Box<dyn FnMut(ProgressEvent) + Send> = Box::new(move |ev: ProgressEvent| { let _ = tx2.send(UiProgressEvent::Service(ev)); });
                // Choose per-file encoding if provided; otherwise use global
                let enc_use_owned: Option<String> = match enc_override {
                    Some(s) if !s.is_empty() => Some(s.clone()),
                    _ => enc_opt.clone(),
                };
                let _ = tx.send(UiProgressEvent::FileStart { index: idx + 1, total: selected.len(), path: p.clone() });
                let _ = svc.ingest_file_with_progress_custom(
                    p, hint,
                    enc_use_owned.as_deref(),
                    min, max, cap, merge_min, ps, pp,
                    Some(&cancel), Some(cb)
                );
                if cancel.is_canceled() { let _ = tx.send(UiProgressEvent::Service(ProgressEvent::Canceled)); return; }
                if idx + 1 == selected.len() { /* Finished will arrive from service */ }
            }
        });
    }

    fn poll_ingest_job(&mut self) {
        if let Some(rx) = &self.ingest_rx {
            loop {
                match rx.try_recv() {
                    Ok(ev) => {
                        match ev {
                            UiProgressEvent::FileStart { index, total, path } => {
                                self.ingest_file_idx = index;
                                self.ingest_file_total = total;
                                // Show file name only for brevity
                                let name = std::path::Path::new(&path).file_name().and_then(|s| s.to_str()).unwrap_or("");
                                self.ingest_file_name = name.to_string();
                            }
                            UiProgressEvent::Service(ProgressEvent::Start { total_chunks }) => {
                                self.ingest_total = total_chunks;
                                self.ingest_done = 0;
                                self.ingest_last_batch = 0;
                                if self.ingest_started.is_none() { self.ingest_started = Some(Instant::now()); }
                                self.status = format!("Embedding {} chunks...", total_chunks);
                            }
                            UiProgressEvent::Service(ProgressEvent::EmbedBatch { done, total, batch }) => {
                                self.ingest_done = done;
                                self.ingest_total = total;
                                self.ingest_last_batch = batch;
                                self.status = format!("Embedding: {} / {} (batch {})", done, total, batch);
                            }
                            UiProgressEvent::Service(ProgressEvent::UpsertDb { total }) => {
                                self.status = format!("Upserting into DB ({} chunks)...", total);
                            }
                            UiProgressEvent::Service(ProgressEvent::IndexText { total }) => {
                                self.status = format!("Indexing text ({} chunks)...", total);
                            }
                            UiProgressEvent::Service(ProgressEvent::IndexVector { total }) => {
                                self.status = format!("Indexing vectors ({} chunks)...", total);
                            }
                            UiProgressEvent::Service(ProgressEvent::SaveIndexes) => {
                                self.status = "Saving indexes...".into();
                            }
                            UiProgressEvent::Service(ProgressEvent::Finished { total }) => {
                                // Service emits Finished per file. Only finalize when the last file is done.
                                let is_last_file = self.ingest_file_total == 0 || self.ingest_file_idx >= self.ingest_file_total;
                                if is_last_file {
                                    self.ingest_running = false;
                                    self.ingest_cancel = None;
                                    self.ingest_rx = None;
                                    let secs = self.ingest_started.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
                                    self.status = format!("Ingest finished ({} chunks) in {:.1}s.", total, secs);
                                    self.ingest_started = None;
                                    self.ingest_file_idx = 0; self.ingest_file_total = 0; self.ingest_file_name.clear();
                                    // Tantivy upsert is handled in the service now
                                    self.ingest_doc_key = None;
                                    break;
                                } else {
                                    // Intermediate file finished; keep running for next file.
                                    self.status = format!("Finished file {} / {}", self.ingest_file_idx, self.ingest_file_total);
                                    // Reset per-file chunk counters for clearer next-file progress
                                    self.ingest_done = 0;
                                    self.ingest_total = 0;
                                    self.ingest_last_batch = 0;
                                }
                            }
                            UiProgressEvent::Service(ProgressEvent::Canceled) => {
                                self.ingest_running = false;
                                self.ingest_cancel = None;
                                self.ingest_rx = None;
                                let secs = self.ingest_started.map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
                                self.status = format!("Ingest canceled after {:.1}s.", secs);
                                self.ingest_started = None;
                                self.ingest_file_idx = 0; self.ingest_file_total = 0; self.ingest_file_name.clear();
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

            // Quick action: Build Prompt button (always visible)
            ui.horizontal(|ui| {
                if ui.button("Build Prompt").clicked() {
                    let out = self.render_prompt();
                    self.prompt_rendered = out;
                    self.prompt_popup_visible = true;
                }
                ui.add_space(8.0);
                if ui.button("Import Templates").clicked() {
                    self.import_prompt_templates_via_dialog();
                }
            });

            // Prompt Template editor
            ui.collapsing("Prompt Template", |ui| {
                // Template selection toolbar
                ui.horizontal(|ui| {
                    ui.label("Template Name");
                    let mut chosen = self.selected_prompt.clone().unwrap_or_default();
                    egui::ComboBox::from_id_source("prompt_tpl_select")
                        .selected_text(if chosen.is_empty() { "<unsaved>" } else { &chosen })
                        .show_ui(ui, |ui| {
                            for tpl in &self.prompt_templates {
                                ui.selectable_value(&mut chosen, tpl.name.clone(), tpl.name.clone());
                            }
                        });
                    if self.selected_prompt.as_deref() != Some(&chosen) {
                        if !chosen.is_empty() {
                            self.selected_prompt = Some(chosen.clone());
                            self.apply_prompt_template_by_name(&chosen);
                        }
                    }
                    ui.add_space(8.0);
                    if ui.button("New").clicked() {
                        self.selected_prompt = None;
                        self.prompt_name_edit_mode = true;
                        self.prompt_name_edit = String::from("New Template");
                    }
                    if ui.button("Delete").clicked() { self.delete_selected_prompt_template(); }
                    if ui.button("Save in Config").clicked() {
                        // Upsert current editor as a template (under selected name or ask name if none)
                        if self.selected_prompt.is_none() && self.prompt_name_edit.trim().is_empty() {
                            // Trigger name input first time for unsaved
                            self.prompt_name_edit_mode = true;
                            self.prompt_name_edit = String::from("New Template");
                        } else {
                            self.save_current_prompt_template(false);
                            self.save_config_overwrite_via_dialog();
                        }
                    }
                });
                if self.prompt_name_edit_mode {
                    ui.horizontal(|ui| {
                        ui.label("Name");
                        ui.add(TextEdit::singleline(&mut self.prompt_name_edit).desired_width(240.0));
                        if ui.button("OK").clicked() { self.save_current_prompt_template(true); self.prompt_name_edit_mode = false; }
                        if ui.button("Cancel").clicked() { self.prompt_name_edit_mode = false; }
                    });
                }

                // Settings: item count and context window
                ui.horizontal(|ui| {
                    ui.label("Items");
                    ui.add(DragValue::new(&mut self.prompt_items_count).clamp_range(1..=1000));
                    ui.add_space(12.0);
                    ui.label("prev");
                    ui.add(DragValue::new(&mut self.prompt_prev).clamp_range(0..=10));
                    ui.label("next");
                    ui.add(DragValue::new(&mut self.prompt_next).clamp_range(0..=10));
                });
                ui.add_space(6.0);
                ui.label("Header template");
                ui.add(TextEdit::multiline(&mut self.prompt_header_tmpl).desired_rows(3).desired_width(700.0));
                ui.add_space(4.0);
                ui.label("Item template");
                ui.add(TextEdit::multiline(&mut self.prompt_item_tmpl).desired_rows(2).desired_width(700.0));
                ui.add_space(4.0);
                ui.label("Footer template");
                ui.add(TextEdit::multiline(&mut self.prompt_footer_tmpl).desired_rows(3).desired_width(700.0));
                ui.add_space(8.0);
            });

            ui.separator();
            ui.push_id("results_table", |ui| {
                egui::ScrollArea::horizontal().id_source("results_table_h").show(ui, |ui| {
                    let results_snapshot = self.results.clone();
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
                            header.col(|ui| {
                                            let active = matches!(self.ingest_sort_key, IngestSortKey::File);
                                            let arrow = if active { if self.ingest_sort_asc { " ▲" } else { " ▼" } } else { "" };
                                            if ui.button(format!("File{}", arrow)).clicked() {
                                                if self.ingest_sort_key != IngestSortKey::File { self.ingest_sort_key = IngestSortKey::File; self.ingest_sort_asc = true; } else if self.ingest_sort_asc { self.ingest_sort_asc = false; } else { self.ingest_sort_key = IngestSortKey::Default; self.ingest_sort_asc = true; }
                                                { let base = self.ingest_folder_path.clone(); let abs = self.ingest_show_abs_paths; self.apply_ingest_sort_with(base.as_str(), abs) };
                                            }
                                        });
                            header.col(|ui| { ui.label("Page"); });
                            if show_tv { header.col(|ui| { ui.label("TV"); }); }
                            if show_tv_and { header.col(|ui| { ui.label("TV(AND)"); }); }
                            if show_tv_or { header.col(|ui| { ui.label("TV(OR)"); }); }
                            if show_vec { header.col(|ui| { ui.label("VEC"); }); }
                            header.col(|ui| { ui.label("Text"); });
                        })
                        .body(|mut body| {
                            for (i, row) in results_snapshot.iter().enumerate() {
                                body.row(20.0, |mut row_ui| {
                                    row_ui.col(|ui| { ui.label(format!("{}", i+1)); });
                                    // Make file and page clickable to select the row
                                    row_ui.col(|ui| {
                                        if ui.link(&row.file).clicked() {
                                            self.selected_cid = Some(row.cid.clone());
                                            self.selected_text = row.text_full.clone();
                                            self.selected_display = format!("{} {}", &row.file, if row.page.is_empty() { String::new() } else { row.page.clone() });
                                            self.selected_source_path = Some(row.file_path.clone());
                                            self.selected_base_cid = Some(row.cid.clone());
                                            self.selected_base_text = row.text_full.clone();
                                            self.selected_base_display = self.selected_display.clone();
                                            self.selected_base_source_path = Some(row.file_path.clone());
                                            self.rebuild_context_window_initial();
                                        }
                                    });
                                    row_ui.col(|ui| {
                                        if !row.page.is_empty() {
                                            if ui.link(&row.page).clicked() {
                                                self.selected_cid = Some(row.cid.clone());
                                                self.selected_text = row.text_full.clone();
                                                self.selected_display = format!("{} {}", &row.file, &row.page);
                                                self.selected_source_path = Some(row.file_path.clone());
                                                self.selected_base_cid = Some(row.cid.clone());
                                                self.selected_base_text = row.text_full.clone();
                                                self.selected_base_display = self.selected_display.clone();
                                                self.selected_base_source_path = Some(row.file_path.clone());
                                                self.rebuild_context_window_initial();
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
                                                self.selected_source_path = Some(row.file_path.clone());
                                                self.selected_base_cid = Some(row.cid.clone());
                                                self.selected_base_text = row.text_full.clone();
                                                self.selected_base_display = self.selected_display.clone();
                                                self.selected_base_source_path = Some(row.file_path.clone());
                                                self.rebuild_context_window_initial();
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
                // Place open actions between the header and the text content
                // Navigation: prev / back-to-base / next
                ui.horizontal(|ui| {
                    if ui.button("<= add prev chunk").clicked() { self.expand_context_prev(); }
                    let can_back = self.selected_base_cid.is_some() && (self.selected_base_cid != self.selected_cid || self.context_expanded);
                    if ui.add_enabled(can_back, Button::new("reset")).clicked() { self.navigate_back_to_base(); }
                    if ui.button("add next chunk =>").clicked() { self.expand_context_next(); }
                });
                ui.add_space(4.0);
                if let Some(path) = &self.selected_source_path {
                    let (is_local, disp) = normalize_local_path_display(path);
                    ui.horizontal(|ui| {
                        let btn_open = ui.add_enabled(is_local, Button::new("Open file"));
                        if btn_open.clicked() && is_local {
                            if let Some(p) = normalize_local_path(path) { let _ = open_in_os(&p); }
                        }
                        let btn_folder = ui.add_enabled(is_local, Button::new("Open folder"));
                        if btn_folder.clicked() && is_local {
                            if let Some(p) = normalize_local_path(path) { let _ = open_in_os_folder(&p); }
                        }
                        if is_local { ui.monospace(disp); }
                    });
                }
                // Detail pane height: 150px
                ScrollArea::vertical().max_height(150.0).id_source("selected_scroll").show(ui, |ui| {
                    let mut job = egui::text::LayoutJob::default();
                    job.wrap.max_width = ui.available_width();
                    let normal_color = ui.visuals().text_color();
                    let weak_color = ui.visuals().weak_text_color();
                    for (i, seg) in self.context_chunks.iter().enumerate() {
                        let mut fmt = egui::text::TextFormat::default();
                        fmt.color = if seg.is_base { normal_color } else { weak_color };
                        fmt.italics = !seg.is_base;
                        job.append(&seg.text, 0.0, fmt);
                        if i + 1 < self.context_chunks.len() {
                            let mut sep_fmt = egui::text::TextFormat::default();
                            sep_fmt.color = weak_color;
                            sep_fmt.italics = true;
                            job.append("\n窶ｦ\n", 0.0, sep_fmt);
                        }
                    }
                    ui.add(egui::Label::new(egui::WidgetText::LayoutJob(job)));
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
                if flat.chars().count() > 80 { text_preview.push('\u{2026}'); }
                self.results.push(HitRow { cid: rec.chunk_id.0, file, file_path: rec.source_uri.clone(), page, text_preview, text_full: rec.text, tv: sc_tv, tv_and: sc_and, tv_or: sc_or, vec: sc_vec });
            }
        }
        self.status = format!("Results: {}", self.results.len());
    }

    fn render_prompt(&self) -> String {
        // Expand header
        let header = expand_template(&self.prompt_header_tmpl, |name, transform, arg| {
            match name {
                "Query" => match transform {
                    Some("escape_json") => escape_json_str(&self.query),
                    Some("snippet") => {
                        let n = arg.and_then(|s| s.parse::<usize>().ok()).unwrap_or(200);
                        snippet_chars(&self.query, n)
                    }
                    Some("snippet_json") => {
                        let n = arg.and_then(|s| s.parse::<usize>().ok()).unwrap_or(200);
                        escape_json_str(&snippet_chars(&self.query, n))
                    }
                    _ => self.query.clone(),
                },
                "TopK" => self.top_k.to_string(),
                _ => String::new(),
            }
        });

        // Expand items
        // Open repo once for optional context expansion
        let repo_opt = chunking_store::sqlite_repo::SqliteRepo::open(self.db_path.trim()).ok();

        let mut body = String::new();
        let item_count = self.prompt_items_count.max(1).min(self.results.len());
        for (i, row) in self.results.iter().take(item_count).enumerate() {
            let item = expand_template(&self.prompt_item_tmpl, |name, transform, arg| {
                match name {
                    "Rank" => (i + 1).to_string(),
                    "File" => row.file.clone(),
                    "Page" => row.page.clone(),
                    "CID" => row.cid.clone(),
                    "SourceUri" => row.file_path.clone(),
                    "Comma" => if i + 1 < item_count { ",".to_string() } else { String::new() },
                    "Text" => match transform {
                        Some("snippet") => {
                            let n = arg.and_then(|s| s.parse::<usize>().ok()).unwrap_or(200);
                            let combined = build_text_with_context_repo(repo_opt.as_ref(), &row.cid, &row.text_full, self.prompt_prev, self.prompt_next);
                            snippet_chars(&combined, n)
                        }
                        Some("escape_json") => {
                            escape_json_str(&build_text_with_context_repo(repo_opt.as_ref(), &row.cid, &row.text_full, self.prompt_prev, self.prompt_next))
                        }
                        Some("snippet_json") => {
                            let n = arg.and_then(|s| s.parse::<usize>().ok()).unwrap_or(200);
                            let combined = build_text_with_context_repo(repo_opt.as_ref(), &row.cid, &row.text_full, self.prompt_prev, self.prompt_next);
                            escape_json_str(&snippet_chars(&combined, n))
                        }
                        _ => build_text_with_context_repo(repo_opt.as_ref(), &row.cid, &row.text_full, self.prompt_prev, self.prompt_next),
                    },
                    _ => String::new(),
                }
            });
            body.push_str(&item);
            if !item.ends_with('\n') { body.push('\n'); }
        }

        // Expand footer
        let footer = expand_template(&self.prompt_footer_tmpl, |name, transform, arg| {
            match name {
                "Query" => match transform {
                    Some("escape_json") => escape_json_str(&self.query),
                    Some("snippet") => {
                        let n = arg.and_then(|s| s.parse::<usize>().ok()).unwrap_or(200);
                        snippet_chars(&self.query, n)
                    }
                    Some("snippet_json") => {
                        let n = arg.and_then(|s| s.parse::<usize>().ok()).unwrap_or(200);
                        escape_json_str(&snippet_chars(&self.query, n))
                    }
                    _ => self.query.clone(),
                },
                "TopK" => self.top_k.to_string(),
                _ => String::new(),
            }
        });

        let mut out = String::new();
        out.push_str(&header);
        if !header.ends_with('\n') { out.push('\n'); }
        out.push_str(&body);
        if !body.ends_with('\n') { out.push('\n'); }
        out.push_str(&footer);
        if !out.ends_with('\n') { out.push('\n'); }
        out
    }

    fn ui_prompt_window(&mut self, ctx: &egui::Context) {
        if !self.prompt_popup_visible { return; }
        let mut open = self.prompt_popup_visible;
        let mut request_close = false;
        egui::Window::new("Prompt Preview")
            .open(&mut open)
            .collapsible(false)
            .default_width(840.0)
            .default_height(600.0)
            .default_pos(egui::pos2(60.0, 60.0))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Copy").clicked() {
                        ui.output_mut(|o| o.copied_text = self.prompt_rendered.clone());
                        self.status = "Prompt copied to clipboard".into();
                    }
                    let close_btn = Button::new(egui::RichText::new("Close").color(egui::Color32::RED).strong());
                    if ui.add(close_btn).clicked() { request_close = true; }
                });
                ui.separator();
                if ui.input(|i| i.key_pressed(egui::Key::Escape)) { request_close = true; }
                ScrollArea::vertical().id_source("prompt_scroll").show(ui, |ui| {
                    ui.add(TextEdit::multiline(&mut self.prompt_rendered).desired_rows(28).desired_width(ui.available_width()));
                });
            });
        self.prompt_popup_visible = open && !request_close;
    }

    fn apply_prompt_template_by_name(&mut self, name: &str) {
        if let Some(t) = self.prompt_templates.iter().find(|t| t.name == name) {
            self.prompt_header_tmpl = t.header.clone();
            self.prompt_item_tmpl = t.item.clone();
            self.prompt_footer_tmpl = t.footer.clone();
            self.prompt_items_count = t.items;
            self.prompt_prev = t.prev;
            self.prompt_next = t.next;
        }
    }

    fn save_current_prompt_template(&mut self, force_new_name: bool) {
        let mut name = self.selected_prompt.clone().unwrap_or_default();
        if force_new_name || name.is_empty() { name = self.prompt_name_edit.trim().to_string(); }
        if name.is_empty() { self.status = "Template name is empty".into(); return; }
        // Build the template object from current editors and settings
        let tpl = PromptTemplate {
            name: name.clone(),
            header: self.prompt_header_tmpl.clone(),
            item: self.prompt_item_tmpl.clone(),
            footer: self.prompt_footer_tmpl.clone(),
            items: self.prompt_items_count,
            prev: self.prompt_prev,
            next: self.prompt_next,
        };
        // Upsert by name
        if let Some(pos) = self.prompt_templates.iter().position(|t| t.name == name) {
            self.prompt_templates[pos] = tpl;
        } else {
            self.prompt_templates.push(tpl);
        }
        self.selected_prompt = Some(name);
        self.status = "Prompt template saved".into();
    }

    fn delete_selected_prompt_template(&mut self) {
        if let Some(name) = self.selected_prompt.clone() {
            if let Some(pos) = self.prompt_templates.iter().position(|t| t.name == name) {
                self.prompt_templates.remove(pos);
                self.selected_prompt = None;
                self.status = "Prompt template deleted".into();
                return;
            }
        }
        self.status = "No template selected".into();
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
    if it.next().is_some() { format!("{}窶ｦ", truncated.replace(['\n', '\r', '\t'], " ")) } else { truncated.replace(['\n', '\r', '\t'], " ") }
}

fn escape_tabs(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch { '\t' => out.push_str("\\t"), '\r' => { /* skip */ }, _ => out.push(ch) }
    }
    out
}

// Display helper: relative to root unless absolute requested or strip fails
fn display_path_with_root(path: &str, root: &str, absolute: bool) -> String {
    if absolute || root.is_empty() { return path.to_string(); }
    use std::path::Path;
    let p = Path::new(path);
    let r = Path::new(root);
    if let Ok(rel) = p.strip_prefix(r) {
        let mut s = rel.display().to_string();
        // Trim a leading separator for aesthetics
        if s.starts_with('\\') || s.starts_with('/') {
            s.remove(0);
        }
        s
    } else {
        path.to_string()
    }
}

// Format SystemTime to yyyy/mm/dd for table display
fn format_ymd(st: std::time::SystemTime) -> String {
    let dt: chrono::DateTime<chrono::Local> = st.into();
    dt.format("%Y/%m/%d").to_string()
}
// Decode bytes with a given encoding keyword used by the GUI.
fn decode_bytes_with_encoding(bytes: &[u8], enc: &str) -> String {
    let enc = enc.to_ascii_lowercase();
    match enc.as_str() {
        "auto" => String::from_utf8_lossy(bytes).to_string(),
        "utf-8" | "utf8" => String::from_utf8_lossy(bytes).to_string(),
        "shift_jis" | "sjis" | "cp932" | "windows-31j" => {
            let (cow, _, _) = encoding_rs::SHIFT_JIS.decode(bytes); cow.into_owned()
        }
        "windows-1252" | "cp1252" => { let (cow, _, _) = encoding_rs::WINDOWS_1252.decode(bytes); cow.into_owned() }
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
        _ => String::from_utf8_lossy(bytes).to_string(),
    }
}

// Load a small preview of a file with the specified encoding, returning a truncated single-line sample.
fn preview_text_for_file(path: &str, enc: &str, max_bytes: usize, max_chars: usize) -> Option<String> {
    use std::fs::File;
    use std::io::Read;
    let mut f = File::open(path).ok()?;
    let mut buf = vec![0u8; max_bytes.max(1)];
    let n = f.read(&mut buf).ok()?;
    buf.truncate(n);
    let mut text = decode_bytes_with_encoding(&buf, enc);
    // normalize CRLF
    text = text.replace('\r', "");
    Some(truncate_for_preview(&text, max_chars))
}

// Compute SHA-256 hex digest of a file path (streaming).
fn sha256_hex_file(path: &str) -> Option<String> {
    use std::fs::File;
    use std::io::Read;
    use sha2::Digest;
    let mut f = File::open(path).ok()?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = match f.read(&mut buf) { Ok(n) => n, Err(_) => return None };
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Some(to_hex(&digest))
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes { s.push_str(&format!("{:02x}", b)); }
    s
}

// Expand placeholders in a template string.
// Supported forms: <<Name>>, <<Name:transform>>, <<Name:transform(arg)>>
// Resolver receives (name, transform, arg) and returns replacement text.
fn expand_template<F>(tmpl: &str, mut resolver: F) -> String
where
    F: FnMut(&str, Option<&str>, Option<&str>) -> String,
{
    let mut out = String::with_capacity(tmpl.len());
    let mut rest = tmpl;
    loop {
        match rest.find("<<") {
            Some(start) => {
                // Push the prefix as-is (preserves UTF-8)
                out.push_str(&rest[..start]);
                let after = &rest[start + 2..];
                if let Some(end_rel) = after.find(">>") {
                    let token = &after[..end_rel];
                    let (name, transform, arg) = parse_token(token);
                    let rep = resolver(name, transform.as_deref(), arg.as_deref());
                    out.push_str(&rep);
                    rest = &after[end_rel + 2..];
                } else {
                    // No closing, push remainder and break
                    out.push_str(rest);
                    break;
                }
            }
            None => {
                out.push_str(rest);
                break;
            }
        }
    }
    out
}

fn parse_token(token: &str) -> (
    &str,                // name
    Option<String>,      // transform
    Option<String>,      // arg
) {
    if let Some(colon) = token.find(':') {
        let (name, rest) = token.split_at(colon);
        let rest = &rest[1..]; // skip ':'
        if let Some(lp) = rest.find('(') {
            let (tf, tail) = rest.split_at(lp);
            if tail.ends_with(')') && tail.len() >= 2 {
                let arg = &tail[1..tail.len() - 1];
                return (name.trim(), Some(tf.trim().to_string()), Some(arg.trim().to_string()));
            }
            return (name.trim(), Some(tf.trim().to_string()), None);
        } else {
            return (name.trim(), Some(rest.trim().to_string()), None);
        }
    }
    (token.trim(), None, None)
}

fn snippet_chars(s: &str, n: usize) -> String {
    if n == 0 { return String::new(); }
    let mut it = s.chars();
    let taken: String = it.by_ref().take(n).collect();
    if it.next().is_some() { format!("{}窶ｦ", taken) } else { taken }
}

fn build_text_with_context_repo(
    repo_opt: Option<&chunking_store::sqlite_repo::SqliteRepo>,
    base_cid: &str,
    base_text: &str,
    prev: usize,
    next: usize,
) -> String {
    if repo_opt.is_none() || (prev == 0 && next == 0) {
        return base_text.to_string();
    }
    let repo = repo_opt.unwrap();
    use chunk_model::ChunkId;
    let mut parts: Vec<String> = Vec::new();
    // Collect prev
    if prev > 0 {
        let mut cur = ChunkId(base_cid.to_string());
        let mut prev_parts: Vec<String> = Vec::new();
        for _ in 0..prev {
            match repo.get_neighbor_chunks(&cur) {
                Ok((p, _)) => {
                    if let Some(pr) = p { prev_parts.push(pr.text.clone()); cur = pr.chunk_id; } else { break; }
                }
                Err(_) => break,
            }
        }
        prev_parts.reverse();
        parts.extend(prev_parts);
    }
    // Base
    parts.push(base_text.to_string());
    // Collect next
    if next > 0 {
        let mut cur = ChunkId(base_cid.to_string());
        for _ in 0..next {
            match repo.get_neighbor_chunks(&cur) {
                Ok((_, nopt)) => {
                    if let Some(nx) = nopt { parts.push(nx.text.clone()); cur = nx.chunk_id; } else { break; }
                }
                Err(_) => break,
            }
        }
    }
    parts.join("\n")
}

fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                let code = c as u32;
                out.push_str(&format!("\\u{:04X}", code));
            }
            _ => out.push(ch),
        }
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

// Normalize a potentially URI-formatted path to a local filesystem path when possible.
fn normalize_local_path(uri: &str) -> Option<String> {
    let mut out: String;
    if uri.starts_with("file://") {
        let mut p = uri.trim_start_matches("file://");
        if cfg!(windows) {
            // Strip leading slash from file:///C:/...
            if p.starts_with('/') { p = &p[1..]; }
        }
        out = p.to_string();
    } else if uri.contains("://") {
        return None;
    } else {
        out = uri.to_string();
    }
    if cfg!(windows) {
        // Best-effort: convert forward slashes and trim surrounding quotes if any
        if out.starts_with('"') && out.ends_with('"') && out.len() >= 2 {
            out = out.trim_matches('"').to_string();
        }
        out = out.replace('/', "\\");
    }
    Some(out)
}

fn normalize_local_path_display(uri: &str) -> (bool, String) {
    match normalize_local_path(uri) {
        Some(p) => (true, p),
        None => (false, String::new()),
    }
}

fn open_in_os(path: &str) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        use std::process::Command;
        // Use explorer to open with associated app; do not add extra quotes
        Command::new("explorer").arg(path).spawn().map_err(|e| e.to_string())?;
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        Command::new("open").arg(path).spawn().map_err(|e| e.to_string())?;
        return Ok(());
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        use std::process::Command;
        Command::new("xdg-open").arg(path).spawn().map_err(|e| e.to_string())?;
        return Ok(());
    }
}

fn open_in_os_folder(path: &str) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        use std::process::Command;
        // Reveal the file in Explorer
        Command::new("explorer").args(["/select,", path]).spawn().map_err(|e| e.to_string())?;
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        // Reveal in Finder
        Command::new("open").args(["-R", path]).spawn().map_err(|e| e.to_string())?;
        return Ok(());
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        use std::process::Command;
        let parent = std::path::Path::new(path).parent().map(|p| p.to_path_buf()).ok_or_else(|| "no parent".to_string())?;
        Command::new("xdg-open").arg(parent).spawn().map_err(|e| e.to_string())?;
        return Ok(());
    }
}














impl AppState {
    fn apply_files_sort(&mut self) {
        let asc = self.files_sort_asc;
        let key = self.files_sort_key;
        self.files.sort_by(|a, b| {
            use std::cmp::Ordering;
            let ord: Ordering = match key {
                FilesSortKey::Default => {
                    let ai = self.files_default_ord.get(&a.doc_id.0).copied().unwrap_or(usize::MAX);
                    let bi = self.files_default_ord.get(&b.doc_id.0).copied().unwrap_or(usize::MAX);
                    ai.cmp(&bi)
                }
                FilesSortKey::File => a.source_uri.cmp(&b.source_uri),
                FilesSortKey::Size => a.file_size_bytes.unwrap_or(0).cmp(&b.file_size_bytes.unwrap_or(0)),
                FilesSortKey::Pages => a.page_count.unwrap_or(0).cmp(&b.page_count.unwrap_or(0)),
                FilesSortKey::Chunks => a.chunk_count.unwrap_or(0).cmp(&b.chunk_count.unwrap_or(0)),
                FilesSortKey::Updated => a.updated_at_meta.as_deref().unwrap_or("").cmp(b.updated_at_meta.as_deref().unwrap_or("")),
                FilesSortKey::Author => a.author_guess.as_deref().unwrap_or("").cmp(b.author_guess.as_deref().unwrap_or("")),
                FilesSortKey::Inserted => a.extracted_at.cmp(&b.extracted_at),
            };
            if asc || matches!(key, FilesSortKey::Default) { ord } else { ord.reverse() }
        });
    }
    fn apply_ingest_sort_with(&mut self, base_root: &str, show_abs: bool) {
        let asc = self.ingest_sort_asc;
        let key = self.ingest_sort_key;
        let make_disp = |p: &str| display_path_with_root(p, base_root, show_abs);
        self.ingest_files.sort_by(|a, b| {
            let ord = match key {
                IngestSortKey::Default => a.ordinal.cmp(&b.ordinal),
                IngestSortKey::File => make_disp(&a.path).cmp(&make_disp(&b.path)),
                IngestSortKey::Size => a.size.cmp(&b.size),
                IngestSortKey::Date => a.modified_ymd.as_deref().unwrap_or("").cmp(b.modified_ymd.as_deref().unwrap_or("")),
                IngestSortKey::Preview => a.preview_cached_text.as_deref().unwrap_or("").cmp(b.preview_cached_text.as_deref().unwrap_or("")),
            };
            if asc || matches!(key, IngestSortKey::Default) { ord } else { ord.reverse() }
        });
    }
}










fn format_ts_local_short(s: &str) -> String {
    // Expect RFC3339/ISO-8601; fallback to original string on parse failure
    if s.is_empty() || s == "-" { return String::from(s); }
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        let local_dt = dt.with_timezone(&chrono::Local);
        return local_dt.format("%Y/%m/%d %H:%M").to_string();
    }
    // Try a common alternative without timezone
    if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        let dt = chrono::Local.from_local_datetime(&ndt).single();
        if let Some(ldt) = dt { return ldt.format("%Y/%m/%d %H:%M").to_string(); }
    }
    s.to_string()
}





