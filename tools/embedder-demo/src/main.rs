use std::fs;
use std::env;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Instant;
use std::path::PathBuf;

use calamine::{open_workbook_auto, Reader};
use csv::{ReaderBuilder, WriterBuilder};
use eframe::egui::{self, Button, CentralPanel, ComboBox, ScrollArea, TextEdit, CollapsingHeader, Spinner};
use eframe::{App, CreationContext, Frame, NativeOptions};
use embedding_provider::config::{default_stdio_config, ONNX_STDIO_DEFAULTS};
use embedding_provider::embedder::{Embedder, OnnxStdIoConfig, OnnxStdIoEmbedder};
use rfd::FileDialog;
use rust_xlsxwriter::{Workbook, XlsxError};
use encoding_rs::{Encoding, SHIFT_JIS, UTF_8};

fn main() -> eframe::Result<()> {
    let options = NativeOptions::default();
    eframe::run_native(
        "Embedder Demo",
        options,
        Box::new(|cc| Box::new(DemoApp::new(cc))),
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveTab {
    Text,
    Excel,
    Csv,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CsvEncoding {
    Utf8,
    ShiftJis,
}
impl CsvEncoding {
    fn label(self) -> &'static str {
        match self {
            CsvEncoding::Utf8 => "UTF-8",
            CsvEncoding::ShiftJis => "Shift_JIS",
        }
    }

    fn encoding(self) -> &'static Encoding {
        match self {
            CsvEncoding::Utf8 => UTF_8,
            CsvEncoding::ShiftJis => SHIFT_JIS,
        }
    }
}

struct DemoApp {
    model_path: String,
    tokenizer_path: String,
    runtime_path: String,
    embedding_dimension: String,
    max_tokens: String,
    status: String,
    input_text: String,
    preview_text: String,
    full_vector_text: String,
    embedder: Option<OnnxStdIoEmbedder>,
    // Excel
    input_excel_path: String,
    output_excel_path: String,
    excel_log: String,
    // CSV
    input_csv_path: String,
    output_csv_path: String,
    csv_log: String,
    csv_encoding: CsvEncoding,
    has_header: bool,
    // Tab
    active_tab: ActiveTab,
    // Async model init
    model_task: Option<ModelInitTask>,
    pending_action: Option<PendingAction>,
    // Async Excel job
    excel_job: Option<JobHandle>,
    // Async CSV job
    csv_job: Option<JobHandle>,
    // Batch size for Excel embedding
    excel_batch_size: usize,
}

struct ModelInitTask {
    rx: Receiver<Result<OnnxStdIoEmbedder, String>>,
    started: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingAction {
    TextEmbed,
    ExcelEmbed,
    CsvEmbed,
}

struct JobHandle {
    rx: Receiver<JobEvent>,
    cancel: Arc<AtomicBool>,
    output_path: String,
    total: usize,
    done: usize,
    started: Instant,
}

enum JobEvent {
    Progress { done: usize, total: usize },
    Finished { rows: usize, dimension: usize, embedder: OnnxStdIoEmbedder },
    Failed { error: String, embedder: OnnxStdIoEmbedder },
}
impl DemoApp {
    fn new(cc: &CreationContext<'_>) -> Self {
        // Install CJK (Japanese) fallback fonts so that UI can render Japanese text.
        install_japanese_fallback_fonts(&cc.egui_ctx);
        let defaults = default_stdio_config();

        // Default testdata paths (if present locally)
        let td = small_testdata_dir();
        let excel_default = td.join(DEFAULT_EXCEL_FILE).display().to_string();
        let csv_default = td
            .join(DEFAULT_CSV_FILE)
            .display()
            .to_string();

        // Default outputs for samples: write into crate-local output/ with same filenames
        let outdir = demo_output_dir();
        let excel_out_default = outdir.join(DEFAULT_EXCEL_FILE).display().to_string();
        let csv_out_default = outdir.join(DEFAULT_CSV_FILE).display().to_string();

        Self {
            model_path: defaults.model_path.display().to_string(),
            tokenizer_path: defaults.tokenizer_path.display().to_string(),
            runtime_path: defaults.runtime_library_path.display().to_string(),
            embedding_dimension: ONNX_STDIO_DEFAULTS.embedding_dimension.to_string(),
            max_tokens: ONNX_STDIO_DEFAULTS.max_input_tokens.to_string(),
            status: "Ready".into(),
            input_text: "sample text for embedding".into(),
            preview_text: String::new(),
            full_vector_text: String::new(),
            embedder: None,
            input_excel_path: excel_default,
            output_excel_path: excel_out_default,
            excel_log: String::new(),
            input_csv_path: csv_default,
            output_csv_path: csv_out_default,
            csv_log: String::new(),
            csv_encoding: CsvEncoding::Utf8,
            has_header: true,
            active_tab: ActiveTab::Text,
            model_task: None,
            pending_action: None,
            excel_job: None,
            csv_job: None,
            excel_batch_size: 64,
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
        if self.model_task.is_some() {
            return;
        }
        let cfg = match self.build_config() {
            Ok(c) => c,
            Err(err) => {
                self.status = err;
                return;
            }
        };
        let (tx, rx) = mpsc::channel();
        let suffix = if context_hint.is_empty() { String::new() } else { format!(" ({context_hint})") };
        self.status = format!("Initializing model{}...", suffix);
        thread::spawn(move || {
            let res = OnnxStdIoEmbedder::new(cfg).map_err(|e| format!("{e}"));
            let _ = tx.send(res);
        });
        self.model_task = Some(ModelInitTask { rx, started: Instant::now() });
    }

    fn queue_or_run_text_embed(&mut self) {
        if self.embedder.is_some() {
            self.run_embed();
        } else {
            self.pending_action = Some(PendingAction::TextEmbed);
            if self.model_task.is_none() {
                self.start_model_init("for text");
            }
            self.status = "Queued: will embed after model initializes".into();
        }
    }

    fn queue_or_run_excel(&mut self) {
        if self.embedder.is_some() {
            self.embed_excel();
        } else {
            self.pending_action = Some(PendingAction::ExcelEmbed);
            if self.model_task.is_none() {
                self.start_model_init("for Excel");
            }
            self.status = "Queued: will run Excel embedding after model initializes".into();
        }
    }

    fn queue_or_run_csv(&mut self) {
        if self.embedder.is_some() {
            self.embed_csv();
        } else {
            self.pending_action = Some(PendingAction::CsvEmbed);
            if self.model_task.is_none() {
                self.start_model_init("for CSV");
            }
            self.status = "Queued: will run CSV embedding after model initializes".into();
        }
    }

    fn run_embed(&mut self) {
        let Some(embedder) = self.embedder.as_ref() else {
            self.status = "Model not initialized".into();
            return;
        };
        let text = self.input_text.trim();
        if text.is_empty() {
            self.status = "Enter input text".into();
            self.preview_text.clear();
            self.full_vector_text.clear();
            return;
        }
        match embedder.embed(text) {
            Ok(vec) => {
                self.preview_text = format_preview(&vec);
                self.full_vector_text = format_full(&vec);
                self.status = format!("Embedding produced {} values", vec.len());
            }
            Err(e) => {
                self.preview_text.clear();
                self.full_vector_text.clear();
                self.status = format!("Embedding failed: {e}");
            }
        }
    }

    fn ensure_embedder(&mut self) -> bool {
        if self.embedder.is_none() {
            if self.model_task.is_none() {
                self.start_model_init("");
            }
            return false;
        }
        true
    }

    fn embed_excel(&mut self) {
        let input_path_str = self.input_excel_path.trim();
        if input_path_str.is_empty() {
            self.status = "Enter the input workbook path".into();
            return;
        }
        let input_path = PathBuf::from(input_path_str);
        if fs::metadata(&input_path).is_err() {
            self.status = format!("Input workbook `{}` is not accessible", input_path.display());
            return;
        }
        if self.output_excel_path.trim().is_empty() {
            let default_output = derive_default_output_path(&input_path, "xlsx");
            self.output_excel_path = default_output.display().to_string();
        }
        let output_path = PathBuf::from(self.output_excel_path.trim());

        if !self.ensure_embedder() {
            return;
        }
        // Take ownership of the embedder for the background job
        let embedder = match self.embedder.take() {
            Some(e) => e,
            None => return,
        };
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_cl = cancel.clone();
        let input_path_th = input_path.clone();
        let output_path_th = output_path.clone();
        let skip = self.has_header;
        self.status = "Running Excel embedding...".into();
        let batch = self.excel_batch_size.max(1);
        thread::spawn(move || {
            run_excel_embedding_job(embedder, &input_path_th, &output_path_th, skip, cancel_cl, tx, batch);
        });
        self.excel_job = Some(JobHandle {
            rx,
            cancel,
            output_path: output_path.display().to_string(),
            total: 0,
            done: 0,
            started: Instant::now(),
        });
    }

    fn embed_csv(&mut self) {
        let input_path_str = self.input_csv_path.trim();
        if input_path_str.is_empty() {
            self.status = "Enter the input CSV path".into();
            return;
        }
        let input_path = PathBuf::from(input_path_str);
        if fs::metadata(&input_path).is_err() {
            self.status = format!("Input CSV `{}` is not accessible", input_path.display());
            return;
        }
        if self.output_csv_path.trim().is_empty() {
            let default_output = derive_default_output_path(&input_path, "csv");
            self.output_csv_path = default_output.display().to_string();
        }
        let output_path = PathBuf::from(self.output_csv_path.trim());

        if !self.ensure_embedder() { return; }
        // spawn async CSV job
        let embedder = match self.embedder.take() { Some(e) => e, None => return };
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_cl = cancel.clone();
        let input_path_th = input_path.clone();
        let output_path_th = output_path.clone();
        let skip = self.has_header;
        let enc = self.csv_encoding;
        let batch = self.excel_batch_size.max(1);
        self.status = "Running CSV embedding...".into();
        thread::spawn(move || {
            run_csv_embedding_job(embedder, &input_path_th, &output_path_th, enc, skip, cancel_cl, tx, batch);
        });
        self.csv_job = Some(JobHandle { rx, cancel, output_path: output_path.display().to_string(), total: 0, done: 0, started: Instant::now() });
    }
}
impl App for DemoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        CentralPanel::default().show(ctx, |ui| {
            // Make the whole UI scrollable vertically as content grows.
            ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            ui.heading("Embedding Model Configuration");

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
                ui.add(TextEdit::singleline(&mut self.embedding_dimension).desired_width(100.0));
                ui.label("Max tokens:");
                ui.add(TextEdit::singleline(&mut self.max_tokens).desired_width(100.0));
                let init_btn = ui.add_enabled(self.model_task.is_none(), Button::new("Initialize Model"));
                if init_btn.clicked() {
                    self.start_model_init("");
                }
                if self.model_task.is_some() {
                    ui.add(Spinner::new());
                    ui.label("Initializing model...");
                    if ui.add(Button::new("Cancel")).clicked() {
                        self.model_task = None;
                        self.pending_action = None;
                        self.status = "Initialization canceled".into();
                    }
                }
            });

            ui.separator();
            // Poll async model initialization task and update status.
            if let Some(task) = &self.model_task {
                match task.rx.try_recv() {
                    Ok(Ok(embedder)) => {
                        let elapsed = task.started.elapsed().as_secs_f32();
                        self.embedder = Some(embedder);
                        self.status = format!("Model initialized in {:.2}s", elapsed);
                        self.model_task = None;
                        // Auto-run any queued action
                        if let Some(action) = self.pending_action.take() {
                            match action {
                                PendingAction::TextEmbed => self.run_embed(),
                                PendingAction::ExcelEmbed => self.embed_excel(),
                                PendingAction::CsvEmbed => self.embed_csv(),
                            }
                        }
                    }
                    Ok(Err(err)) => {
                        self.status = format!("Failed to initialize: {err}");
                        self.model_task = None;
                    }
                    Err(TryRecvError::Empty) => {
                        // keep waiting
                        ctx.request_repaint();
                    }
                    Err(TryRecvError::Disconnected) => {
                        self.status = "Initialization failed (disconnected)".into();
                        self.model_task = None;
                    }
                }
            }
            ui.label(format!("Status: {}", self.status));

            ui.separator();

            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.active_tab, ActiveTab::Text, "Text");
                ui.selectable_value(&mut self.active_tab, ActiveTab::Excel, "Excel");
                ui.selectable_value(&mut self.active_tab, ActiveTab::Csv, "CSV");
            });

            ui.separator();

            match self.active_tab {
                ActiveTab::Text => {
                    ui.heading("Run Embedding (Text)");
                    ui.add(TextEdit::multiline(&mut self.input_text).desired_rows(4).desired_width(600.0));
                    if ui.add(Button::new("Embed")).clicked() {
                        self.queue_or_run_text_embed();
                    }

                    ui.separator();
                    ui.heading("Preview");
                    ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                        ui.monospace(&self.preview_text);
                    });

                    CollapsingHeader::new("Full Vector")
                        .default_open(false)
                        .show(ui, |ui| {
                            ScrollArea::vertical().max_height(300.0).show(ui, |ui| {
                                ui.monospace(&self.full_vector_text);
                            });
                        });
                }
                ActiveTab::Excel => {
                    ui.heading("Embed from Excel (first sheet, first column)");
                    ui.horizontal(|ui| {
                        ui.label("Input workbook:");
                        ui.add(TextEdit::singleline(&mut self.input_excel_path).desired_width(400.0));
                        if ui.add(Button::new("Browse")).clicked() {
                            let td = small_testdata_dir();
                            if let Some(path) = FileDialog::new()
                                .add_filter("Excel", &["xlsx", "xls"])
                                .set_directory(&td)
                                .set_file_name(DEFAULT_EXCEL_FILE)
                                .pick_file()
                            {
                                self.input_excel_path = path.display().to_string();
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.has_header, "Skip first row (header)");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Output workbook (optional):");
                        ui.add(TextEdit::singleline(&mut self.output_excel_path).desired_width(400.0));
                        if ui.add(Button::new("Save As")).clicked() {
                            if let Some(path) = FileDialog::new().add_filter("Excel", &["xlsx"]).save_file() {
                                self.output_excel_path = path.display().to_string();
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Batch size:");
                        ui.add(egui::Slider::new(&mut self.excel_batch_size, 1..=256).text("rows/batch"));
                    });
                    if ui.add(Button::new("Run Excel Embedding")).clicked() {
                        self.queue_or_run_excel();
                    }
                    let mut excel_end_event: Option<JobEvent> = None;
                    let mut excel_disconnected = false;
                    if let Some(job) = &mut self.excel_job {
                        // Poll progress events
                        loop {
                            match job.rx.try_recv() {
                                Ok(JobEvent::Progress { done, total }) => {
                                    job.done = done;
                                    job.total = total;
                                    self.status = format!("Excel embedding: {}/{}", done, total);
                                }
                                Ok(ev @ JobEvent::Finished { .. }) | Ok(ev @ JobEvent::Failed { .. }) => { excel_end_event = Some(ev); break; }
                                Err(TryRecvError::Empty) => { break; }
                                Err(TryRecvError::Disconnected) => { excel_disconnected = true; break; }
                            }
                        }
                        ui.separator();
                        ui.label(format!("Progress: {}/{}", job.done, job.total));
                        if ui.add(Button::new("Cancel Excel Job")).clicked() {
                            job.cancel.store(true, Ordering::SeqCst);
                            self.status = "Canceling Excel embedding...".into();
                        }
                    }
                    if excel_disconnected {
                        self.status = "Excel embedding failed (disconnected)".into();
                        self.excel_job = None;
                    }
                    if let Some(ev) = excel_end_event {
                        if let Some(job) = &self.excel_job { let _ = job; } // no-op to satisfy borrow checker logic
                        match ev {
                            JobEvent::Finished { rows, dimension, embedder } => {
                                self.embedder = Some(embedder);
                                if let Some(job) = &self.excel_job {
                                    let elapsed = job.started.elapsed().as_secs_f32();
                                    self.status = format!(
                                        "Excel embedding completed in {:.2}s: {} rows written to `{}`",
                                        elapsed, rows, job.output_path
                                    );
                                    self.excel_log = format!(
                                        "Output: {}\nRows processed: {}\nEmbedding dimension: {}\nElapsed: {:.2}s",
                                        job.output_path,
                                        rows,
                                        dimension,
                                        elapsed
                                    );
                                }
                            }
                            JobEvent::Failed { error, embedder } => {
                                self.embedder = Some(embedder);
                                self.status = format!("Excel embedding failed: {error}");
                                self.excel_log = error;
                            }
                            _ => {}
                        }
                        self.excel_job = None;
                    }
                    ui.separator();
                    ui.heading("Excel Log");
                    ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                        ui.monospace(&self.excel_log);
                    });
                }
                ActiveTab::Csv => {
                    ui.heading("Embed from CSV (first column)");
                    ui.horizontal(|ui| {
                        ui.label("Input CSV:");
                        ui.add(TextEdit::singleline(&mut self.input_csv_path).desired_width(400.0));
                        if ui.add(Button::new("Browse")).clicked() {
                            let td = small_testdata_dir();
                            if let Some(path) = FileDialog::new()
                                .add_filter("CSV", &["csv"])    
                                .set_directory(&td)
                                .set_file_name(DEFAULT_CSV_FILE)
                                .pick_file()
                            {
                                self.input_csv_path = path.display().to_string();
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.has_header, "Skip first row (header)");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Output CSV (optional):");
                        ui.add(TextEdit::singleline(&mut self.output_csv_path).desired_width(400.0));
                        if ui.add(Button::new("Save As")).clicked() {
                            if let Some(path) = FileDialog::new().add_filter("CSV", &["csv"]).save_file() {
                                self.output_csv_path = path.display().to_string();
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Encoding:");
                        ComboBox::from_id_source("csv-enc")
                            .selected_text(self.csv_encoding.label())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.csv_encoding, CsvEncoding::Utf8, CsvEncoding::Utf8.label());
                                ui.selectable_value(&mut self.csv_encoding, CsvEncoding::ShiftJis, CsvEncoding::ShiftJis.label());
                            });
                    });
                    if ui.add(Button::new("Run CSV Embedding")).clicked() {
                        self.queue_or_run_csv();
                    }
                    // CSV job progress
                    let mut csv_end_event: Option<JobEvent> = None;
                    let mut csv_disconnected = false;
                    if let Some(job) = &mut self.csv_job {
                        loop {
                            match job.rx.try_recv() {
                                Ok(JobEvent::Progress { done, total }) => {
                                    job.done = done;
                                    job.total = total;
                                    self.status = format!("CSV embedding: {}/{}", done, total);
                                }
                                Ok(ev @ JobEvent::Finished { .. }) | Ok(ev @ JobEvent::Failed { .. }) => { csv_end_event = Some(ev); break; }
                                Err(TryRecvError::Empty) => { break; }
                                Err(TryRecvError::Disconnected) => { csv_disconnected = true; break; }
                            }
                        }
                        ui.separator();
                        ui.label(format!("Progress: {}/{}", job.done, job.total));
                        if ui.add(Button::new("Cancel CSV Job")).clicked() {
                            job.cancel.store(true, Ordering::SeqCst);
                            self.status = "Canceling CSV embedding...".into();
                        }
                    }
                    if csv_disconnected {
                        self.status = "CSV embedding failed (disconnected)".into();
                        self.csv_job = None;
                    }
                    if let Some(ev) = csv_end_event {
                        if let Some(job) = &self.csv_job {
                            match ev {
                                JobEvent::Finished { rows, dimension, embedder } => {
                                    self.embedder = Some(embedder);
                                    let elapsed = job.started.elapsed().as_secs_f32();
                                    self.status = format!("CSV embedding completed in {:.2}s: {} rows written to `{}`", elapsed, rows, job.output_path);
                                    self.csv_log = format!("Output: {}\nRows processed: {}\nEmbedding dimension: {}\nElapsed: {:.2}s", job.output_path, rows, dimension, elapsed);
                                }
                                JobEvent::Failed { error, embedder } => {
                                    self.embedder = Some(embedder);
                                    self.status = format!("CSV embedding failed: {error}");
                                    self.csv_log = error;
                                }
                                _ => {}
                            }
                        }
                        self.csv_job = None;
                    }
                    ui.separator();
                    ui.heading("CSV Log");
                    ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                        ui.monospace(&self.csv_log);
                    });
                }
            }

            });
        });
    }
}

#[allow(dead_code)]
struct BatchStats {
    rows: usize,
    dimension: usize,
}

fn derive_default_output_path(input: &PathBuf, ext: &str) -> PathBuf {
    let parent = input
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(PathBuf::new);
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("embedded");
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let mut candidate = if parent.as_os_str().is_empty() {
        PathBuf::from(format!("{stem}_embed_{timestamp}.{ext}"))
    } else {
        parent.join(format!("{stem}_embed_{timestamp}.{ext}"))
    };
    let mut counter = 1;
    while candidate.exists() {
        let suffix = format!("{stem}_embed_{timestamp}_{counter}.{ext}");
        candidate = if parent.as_os_str().is_empty() {
            PathBuf::from(&suffix)
        } else {
            parent.join(&suffix)
        };
        counter += 1;
    }
    candidate
}

#[allow(dead_code)]
fn embed_excel_file(
    embedder: &dyn Embedder,
    input_path: &std::path::Path,
    output_path: &std::path::Path,
    skip_first_row: bool,
) -> Result<BatchStats, String> {
    let mut workbook = open_workbook_auto(input_path)
        .map_err(|err| format!("failed to open `{}`: {err}", input_path.display()))?;

    let range = workbook
        .worksheet_range_at(0)
        .ok_or_else(|| "workbook does not contain any worksheets".to_string())?
        .map_err(|err| format!("failed to read first worksheet: {err}"))?;

    let mut rows = Vec::new();
    let mut header_label: Option<String> = None;
    for (i, row) in range.rows().enumerate() {
        if i == 0 {
            // Capture original header name if requested, then optionally skip
            if skip_first_row {
                header_label = row
                    .get(0)
                    .map(|cell| cell.to_string())
                    .and_then(|s| {
                        let t = s.trim().to_string();
                        if t.is_empty() { None } else { Some(t) }
                    });
                continue;
            }
        }
        let text = row
            .get(0)
            .map(|cell| cell.to_string())
            .unwrap_or_default()
            .trim()
            .to_owned();
        rows.push(text);
    }

    if rows.is_empty() {
        return Err("input workbook contains no rows".into());
    }

    let text_refs: Vec<&str> = rows.iter().map(String::as_str).collect();
    let embeddings = embedder
        .embed_batch(&text_refs)
        .map_err(|err| format!("embedding failed: {err}"))?;

    if embeddings.len() != rows.len() {
        return Err(format!(
            "expected {} embeddings but got {}",
            rows.len(),
            embeddings.len()
        ));
    }

    let dimension = embeddings
        .first()
        .map(|vector| vector.len())
        .unwrap_or_else(|| embedder.info().dimension);

    if dimension == 0 {
        return Err("embedding vectors are empty".into());
    }

    for (idx, vector) in embeddings.iter().enumerate() {
        if vector.len() != dimension {
            return Err(format!(
                "embedding length mismatch at row {} (expected {}, got {})",
                idx + 1,
                dimension,
                vector.len()
            ));
        }
    }

    let mut workbook_out = Workbook::new();
    {
        let worksheet = workbook_out.add_worksheet();

        let header_title = header_label
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("text");
        worksheet
            .write_string(0, 0, header_title)
            .map_err(|err| map_xlsx_error("write header", err))?;

        for col in 0..dimension {
            let header = format!("emb_{col}");
            let col_index = excel_column_index(col + 1)?;
            worksheet
                .write_string(0, col_index, &header)
                .map_err(|err| map_xlsx_error("write header", err))?;
        }

        for (row_idx, (text, vector)) in rows.iter().zip(embeddings.iter()).enumerate() {
            worksheet
                .write_string((row_idx + 1) as u32, 0, text)
                .map_err(|err| map_xlsx_error("write text cell", err))?;
            for (col_idx, value) in vector.iter().enumerate() {
                let col_index = excel_column_index(col_idx + 1)?;
                worksheet
                    .write_number((row_idx + 1) as u32, col_index, f64::from(*value))
                    .map_err(|err| {
                        format!(
                            "write embedding at row {}, column {} failed: {err}",
                            row_idx + 1,
                            col_idx + 1
                        )
                    })?;
            }
        }
    }

    workbook_out
        .save(output_path)
        .map_err(|err| map_xlsx_error("save workbook", err))?;

    Ok(BatchStats {
        rows: rows.len(),
        dimension,
    })
}

fn map_xlsx_error(context: &str, err: XlsxError) -> String {
    format!("{context} failed: {err}")
}

fn excel_column_index(index: usize) -> Result<u16, String> {
    u16::try_from(index).map_err(|_| {
        format!(
            "column index {} exceeds supported Excel column limit",
            index
        )
    })
}

fn format_embedding_value(value: f32) -> String {
    format!("{value:.6}")
}

const DEFAULT_EXCEL_FILE: &str = "one_col_10_records_with_header.xlsx";
const DEFAULT_CSV_FILE: &str = "one_col_10_records_with_header_utf8bom.csv";

fn small_testdata_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata").join("small")
}

fn demo_output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("output")
}

fn format_preview(vector: &[f32]) -> String {
    if vector.is_empty() {
        return String::new();
    }
    let max = vector.len().min(256);
    let mut out = String::new();
    for chunk in vector.iter().take(max).collect::<Vec<_>>().chunks(8) {
        let line = chunk
            .iter()
            .map(|v| format!("{:>10.6}", v))
            .collect::<Vec<_>>()
            .join(" ");
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn format_full(vector: &[f32]) -> String {
    if vector.is_empty() {
        return String::new();
    }
    vector
        .iter()
        .enumerate()
        .map(|(i, v)| format!("{i:04}: {:.12}", v))
        .collect::<Vec<_>>()
        .join("\n")
}

fn run_excel_embedding_job(
    embedder: OnnxStdIoEmbedder,
    input_path: &std::path::Path,
    output_path: &std::path::Path,
    skip_first_row: bool,
    cancel: Arc<AtomicBool>,
    tx: mpsc::Sender<JobEvent>,
    batch_size: usize,
) {
    // open workbook
    let mut workbook = match open_workbook_auto(input_path) {
        Ok(wb) => wb,
        Err(e) => {
            let _ = tx.send(JobEvent::Failed { error: format!("failed to open `{}`: {e}", input_path.display()), embedder });
            return;
        }
    };

    let range = match workbook.worksheet_range_at(0) {
        Some(Ok(r)) => r,
        Some(Err(e)) => {
            let _ = tx.send(JobEvent::Failed { error: format!("failed to read first worksheet: {e}"), embedder });
            return;
        }
        None => {
            let _ = tx.send(JobEvent::Failed { error: "workbook does not contain any worksheets".to_string(), embedder });
            return;
        }
    };

    let mut rows: Vec<String> = Vec::new();
    let mut header_label: Option<String> = None;
    for (i, row) in range.rows().enumerate() {
        if i == 0 && skip_first_row {
            header_label = row
                .get(0)
                .map(|cell| cell.to_string())
                .and_then(|s| {
                    let t = s.trim().to_string();
                    if t.is_empty() { None } else { Some(t) }
                });
            continue;
        }
        let text = row
            .get(0)
            .map(|cell| cell.to_string())
            .unwrap_or_default()
            .trim()
            .to_owned();
        if !text.is_empty() {
            rows.push(text);
        }
    }

    if rows.is_empty() {
        let _ = tx.send(JobEvent::Failed { error: "input workbook contains no rows".into(), embedder });
        return;
    }

    let total = rows.len();
    let mut workbook_out = Workbook::new();
    let worksheet = workbook_out.add_worksheet();
    let header_title = header_label
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or("text");
    if let Err(e) = worksheet.write_string(0, 0, header_title) {
        let _ = tx.send(JobEvent::Failed { error: map_xlsx_error("write header", e), embedder });
        return;
    }

    // Determine dimension
    let dim = match embedder.embed(&rows[0]) {
        Ok(v) => v.len(),
        Err(e) => {
            let _ = tx.send(JobEvent::Failed { error: format!("embedding failed: {e}"), embedder });
            return;
        }
    };
    for col in 0..dim {
        let header = format!("emb_{col}");
        let col_index = match excel_column_index(col + 1) {
            Ok(ix) => ix,
            Err(e) => { let _ = tx.send(JobEvent::Failed { error: e, embedder }); return; }
        };
        if let Err(e) = worksheet.write_string(0, col_index, &header) {
            let _ = tx.send(JobEvent::Failed { error: map_xlsx_error("write header", e), embedder });
            return;
        }
    }

    let batch_size = batch_size.max(1);
    let mut done = 0usize;
    while done < total {
        if cancel.load(Ordering::SeqCst) {
            let _ = tx.send(JobEvent::Failed { error: "canceled by user".into(), embedder });
            return;
        }
        let end = usize::min(done + batch_size, total);
        let batch = &rows[done..end];
        let refs: Vec<&str> = batch.iter().map(String::as_str).collect();
        let vectors = match embedder.embed_batch(&refs) {
            Ok(v) => v,
            Err(e) => { let _ = tx.send(JobEvent::Failed { error: format!("embedding failed: {e}"), embedder }); return; }
        };

        for (idx, (text, vec)) in batch.iter().zip(vectors.iter()).enumerate() {
            let r = (done + idx + 1) as u32; // +1 for header
            if let Err(e) = worksheet.write_string(r, 0, text) {
                let _ = tx.send(JobEvent::Failed { error: map_xlsx_error("write text cell", e), embedder });
                return;
            }
            for (col_idx, value) in vec.iter().enumerate() {
                let col_index = match excel_column_index(col_idx + 1) { Ok(ix) => ix, Err(e) => { let _ = tx.send(JobEvent::Failed { error: e, embedder }); return; } };
                if let Err(e) = worksheet.write_number(r, col_index, f64::from(*value)) {
                    let _ = tx.send(JobEvent::Failed { error: map_xlsx_error("write embedding", e), embedder });
                    return;
                }
            }
        }

        done = end;
        let _ = tx.send(JobEvent::Progress { done, total });
    }

    if let Err(e) = workbook_out.save(output_path) {
        let _ = tx.send(JobEvent::Failed { error: map_xlsx_error("save workbook", e), embedder });
        return;
    }

    let _ = tx.send(JobEvent::Finished { rows: rows.len(), dimension: dim, embedder });
}

fn run_csv_embedding_job(
    embedder: OnnxStdIoEmbedder,
    input_path: &std::path::Path,
    output_path: &std::path::Path,
    encoding: CsvEncoding,
    skip_first_row: bool,
    cancel: Arc<AtomicBool>,
    tx: mpsc::Sender<JobEvent>,
    batch_size: usize,
) {
    let codec = encoding.encoding();
    // Read input file
    let bytes = match fs::read(input_path) {
        Ok(b) => b,
        Err(e) => { let _ = tx.send(JobEvent::Failed { error: format!("failed to read `{}`: {e}", input_path.display()), embedder }); return; }
    };
    let (decoded, _, had_decode_errors) = codec.decode(&bytes);
    if had_decode_errors { let _ = tx.send(JobEvent::Failed { error: format!("failed to decode `{}` as {}", input_path.display(), encoding.label()), embedder }); return; }
    let mut csv_input = decoded.into_owned();
    if csv_input.starts_with('\u{feff}') { csv_input.remove(0); }
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(csv_input.as_bytes());

    let mut rows: Vec<String> = Vec::new();
    let mut header_label: Option<String> = None;
    for (index, record) in reader.records().enumerate() {
        let record = match record { Ok(r) => r, Err(e) => { let _ = tx.send(JobEvent::Failed { error: format!("failed to read record {}: {e}", index + 1), embedder }); return; } };
        if index == 0 && skip_first_row {
            header_label = record.get(0).map(|s| s.trim().to_owned()).and_then(|s| if s.is_empty() { None } else { Some(s) });
            continue;
        }
        let text = record.get(0).unwrap_or("").trim().to_owned();
        if !text.is_empty() { rows.push(text); }
    }
    if rows.is_empty() { let _ = tx.send(JobEvent::Failed { error: "input CSV contains no rows".into(), embedder }); return; }

    // Probe embedding dimension
    let dim = match embedder.embed(&rows[0]) { Ok(v) => v.len(), Err(e) => { let _ = tx.send(JobEvent::Failed { error: format!("embedding failed: {e}"), embedder }); return; } };

    // Prepare CSV writer into buffer
    let mut buffer = Vec::new();
    {
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(&mut buffer);
        let mut header = Vec::with_capacity(dim + 1);
        let header_title = header_label.as_deref().filter(|s| !s.trim().is_empty()).unwrap_or("text");
        header.push(header_title.to_string());
        for index in 0..dim { header.push(format!("emb_{index}")); }
        if let Err(e) = writer.write_record(&header) { let _ = tx.send(JobEvent::Failed { error: format!("failed to write CSV header: {e}"), embedder }); return; }

        let total = rows.len();
        let bs = batch_size.max(1);
        let mut done = 0usize;
        while done < total {
            if cancel.load(Ordering::SeqCst) { let _ = tx.send(JobEvent::Failed { error: "canceled by user".into(), embedder }); return; }
            let end = usize::min(done + bs, total);
            let batch = &rows[done..end];
            let refs: Vec<&str> = batch.iter().map(String::as_str).collect();
            let vectors = match embedder.embed_batch(&refs) { Ok(v) => v, Err(e) => { let _ = tx.send(JobEvent::Failed { error: format!("embedding failed: {e}"), embedder }); return; } };
            for (text, vec) in batch.iter().zip(vectors.iter()) {
                let mut record = Vec::with_capacity(vec.len() + 1);
                record.push(text.to_string());
                record.extend(vec.iter().map(|value| format_embedding_value(*value)));
                if let Err(e) = writer.write_record(&record) { let _ = tx.send(JobEvent::Failed { error: format!("failed to write CSV row: {e}"), embedder }); return; }
            }
            done = end;
            let _ = tx.send(JobEvent::Progress { done, total });
        }
        if let Err(e) = writer.flush() { let _ = tx.send(JobEvent::Failed { error: format!("failed to finalize CSV writer: {e}"), embedder }); return; }
    }

    // Encode to target encoding and write to disk
    let csv_utf8 = match String::from_utf8(buffer) { Ok(s) => s, Err(e) => { let _ = tx.send(JobEvent::Failed { error: format!("invalid UTF-8 output: {e}"), embedder }); return; } };
    let (encoded, _, had_encode_errors) = codec.encode(&csv_utf8);
    if had_encode_errors { let _ = tx.send(JobEvent::Failed { error: format!("output contains characters not representable in {}", encoding.label()), embedder }); return; }
    if let Err(e) = fs::write(output_path, encoded.as_ref()) { let _ = tx.send(JobEvent::Failed { error: format!("failed to write `{}`: {e}", output_path.display()), embedder }); return; }

    let _ = tx.send(JobEvent::Finished { rows: rows.len(), dimension: dim, embedder });
}

#[allow(dead_code)]
fn embed_csv_file(
    embedder: &dyn Embedder,
    input_path: &std::path::Path,
    output_path: &std::path::Path,
    encoding: CsvEncoding,
    skip_first_row: bool,
) -> Result<BatchStats, String> {
    let codec = encoding.encoding();

    let bytes = fs::read(input_path)
        .map_err(|err| format!("failed to read `{}`: {err}", input_path.display()))?;
    let (decoded, _, had_decode_errors) = codec.decode(&bytes);
    if had_decode_errors {
        return Err(format!(
            "failed to decode `{}` as {}",
            input_path.display(),
            encoding.label()
        ));
    }
    let mut csv_input = decoded.into_owned();
    if csv_input.starts_with('\u{feff}') {
        csv_input.remove(0);
    }

    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(csv_input.as_bytes());

    let mut rows = Vec::new();
    let mut header_label: Option<String> = None;
    for (index, record) in reader.records().enumerate() {
        let record = record.map_err(|err| format!("failed to read record {}: {err}", index + 1))?;
        if index == 0 && skip_first_row {
            header_label = record.get(0).map(|s| s.trim().to_owned()).and_then(|s| if s.is_empty() { None } else { Some(s) });
            continue;
        }
        let text = record.get(0).unwrap_or("").trim().to_owned();
        rows.push(text);
    }

    if rows.is_empty() {
        return Err("input CSV contains no rows".into());
    }

    drop(reader);

    let text_refs: Vec<&str> = rows.iter().map(String::as_str).collect();
    let embeddings = embedder
        .embed_batch(&text_refs)
        .map_err(|err| format!("embedding failed: {err}"))?;

    if embeddings.len() != rows.len() {
        return Err(format!(
            "expected {} embeddings but got {}",
            rows.len(),
            embeddings.len()
        ));
    }

    let dimension = embeddings
        .first()
        .map(|vector| vector.len())
        .unwrap_or_else(|| embedder.info().dimension);

    if dimension == 0 {
        return Err("embedding vectors are empty".into());
    }

    for (idx, vector) in embeddings.iter().enumerate() {
        if vector.len() != dimension {
            return Err(format!(
                "embedding length mismatch at row {} (expected {}, got {})",
                idx + 1,
                dimension,
                vector.len()
            ));
        }
    }

    let mut buffer = Vec::new();
    {
        let mut writer = WriterBuilder::new()
            .has_headers(false)
            .from_writer(&mut buffer);

        let mut header = Vec::with_capacity(dimension + 1);
        let header_title = header_label.as_deref().filter(|s| !s.trim().is_empty()).unwrap_or("text");
        header.push(header_title.to_string());
        for index in 0..dimension {
            header.push(format!("emb_{index}"));
        }
        writer
            .write_record(&header)
            .map_err(|err| format!("failed to write CSV header: {err}"))?;

        for (text, vector) in rows.iter().zip(embeddings.iter()) {
            let mut record = Vec::with_capacity(vector.len() + 1);
            record.push(text.to_string());
            record.extend(vector.iter().map(|value| format_embedding_value(*value)));
            writer
                .write_record(&record)
                .map_err(|err| format!("failed to write CSV row: {err}"))?;
        }

        writer
            .flush()
            .map_err(|err| format!("failed to finalize CSV writer: {err}"))?;
    }

    let csv_utf8 =
        String::from_utf8(buffer).map_err(|err| format!("invalid UTF-8 output: {err}"))?;
    let (encoded, _, had_encode_errors) = codec.encode(&csv_utf8);
    if had_encode_errors {
        return Err(format!(
            "output contains characters not representable in {}",
            encoding.label()
        ));
    }

    fs::write(output_path, encoded.as_ref())
        .map_err(|err| format!("failed to write `{}`: {err}", output_path.display()))?;

    Ok(BatchStats {
        rows: rows.len(),
        dimension,
    })
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



