use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, atomic::{AtomicBool, Ordering}};

use chrono::Utc;
use chunk_model::{ChunkId, ChunkRecord, DocumentId};
use chunking_store::fts5_index::Fts5Index;
use chunking_store::hnsw_index::HnswIndex;
use chunking_store::orchestrator::{delete_by_filter_orchestrated, ingest_chunks_orchestrated, DeleteReport};
use chunking_store::{ChunkStoreRead, FilterClause, SearchHit, SearchOptions, TextSearcher, VectorSearcher};
use chunking_store::sqlite_repo::SqliteRepo;
use embedding_provider::config::default_stdio_config;
use embedding_provider::embedder::{Embedder, OnnxStdIoConfig, OnnxStdIoEmbedder};

#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error("repo error: {0}")]
    Repo(String),
    #[error("embedder error: {0}")]
    Embed(String),
    #[error("index error: {0}")]
    Index(String),
    #[error("io error: {0}")]
    Io(String),
}

#[derive(Debug, Clone)]
pub struct ServiceConfig {
    pub db_path: PathBuf,
    pub hnsw_dir: Option<PathBuf>,
    pub embedder: OnnxStdIoConfig,
    /// Max number of chunks to embed per batch to control memory usage.
    pub embed_batch_size: usize,
    /// When true, automatically adapts batch size (backoff + bucketing).
    pub embed_auto: bool,
    /// Initial batch size to try in auto mode.
    pub embed_initial_batch: usize,
    /// Minimum batch size to allow in auto mode.
    pub embed_min_batch: usize,
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("target/demo/chunks.db"),
            hnsw_dir: None,
            embedder: default_stdio_config(),
            embed_batch_size: 64,
            embed_auto: true,
            embed_initial_batch: 128,
            embed_min_batch: 8,
        }
    }
}

pub struct HybridService {
    cfg: ServiceConfig,
    embedder: OnnxStdIoEmbedder,
    hnsw: Arc<RwLock<Option<HnswIndex>>>,
    warmed: AtomicBool,
}

/// Cooperative cancellation handle shared across long-running operations.
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);
impl CancelToken {
    pub fn new() -> Self { Self(Arc::new(AtomicBool::new(false))) }
    pub fn cancel(&self) { self.0.store(true, Ordering::Relaxed); }
    pub fn is_canceled(&self) -> bool { self.0.load(Ordering::Relaxed) }
}

/// Progress events emitted during ingestion.
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    Start { total_chunks: usize },
    EmbedBatch { done: usize, total: usize, batch: usize },
    UpsertDb { total: usize },
    IndexText { total: usize },
    IndexVector { total: usize },
    SaveIndexes,
    Finished { total: usize },
    Canceled,
}

impl HybridService {
    pub fn new(cfg: ServiceConfig) -> Result<Self, ServiceError> {
        // Ensure DB dir exists
        if let Some(dir) = cfg.db_path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| ServiceError::Io(e.to_string()))?;
        }
        let embedder = OnnxStdIoEmbedder::new(cfg.embedder.clone())
            .map_err(|e| ServiceError::Embed(e.to_string()))?;
        let svc = Self { cfg, embedder, hnsw: Arc::new(RwLock::new(None)), warmed: AtomicBool::new(false) };
        // Warm up ONNX session once (best-effort)
        let _ = svc.embedder.embed("warmup").map(|_| svc.warmed.store(true, Ordering::Relaxed));
        // Background preload HNSW (if index exists)
        let hdir = svc.hnsw_dir();
        let dim = svc.embedder.info().dimension;
        let cache = Arc::clone(&svc.hnsw);
        std::thread::spawn(move || {
            if Path::new(&hdir).join("map.tsv").exists() {
                if let Ok(h) = HnswIndex::load(&hdir, dim) {
                    if let Ok(mut guard) = cache.write() { *guard = Some(h); }
                }
            }
        });
        Ok(svc)
    }

    /// Returns true when the resident HNSW index is loaded in memory.
    pub fn hnsw_ready(&self) -> bool {
        if let Ok(g) = self.hnsw.read() {
            g.as_ref().is_some()
        } else {
            false
        }
    }

    fn ensure_warm(&self) {
        if !self.warmed.load(Ordering::Relaxed) {
            if self.embedder.embed("warmup").is_ok() {
                self.warmed.store(true, Ordering::Relaxed);
            }
        }
    }

    fn open_repo(&self) -> Result<SqliteRepo, ServiceError> {
        let repo = SqliteRepo::open(&self.cfg.db_path).map_err(|e| ServiceError::Repo(e.to_string()))?;
        let _ = repo.maybe_rebuild_fts();
        Ok(repo)
    }

    fn hnsw_dir(&self) -> PathBuf {
        match &self.cfg.hnsw_dir {
            Some(d) => d.clone(),
            None => derive_hnsw_dir(&self.cfg.db_path),
        }
    }

    /// Ingest pre-built chunks with optional precomputed vectors.
    pub fn ingest_chunks(&self, records: &[ChunkRecord], vectors: Option<&[(ChunkId, Vec<f32>)]>) -> Result<(), ServiceError> {
        if records.is_empty() { return Ok(()); }
        let mut repo = self.open_repo()?;

        // Prepare FTS maintainer (triggers handle it but keep API consistent)
        let fts = Fts5Index::new();
        let text_m: [&dyn chunking_store::TextIndexMaintainer; 1] = [&fts];

        // Prepare/load HNSW
        let hdir = self.hnsw_dir();
        let mut hnsw = if Path::new(&hdir).join("map.tsv").exists() {
            HnswIndex::load(&hdir, self.embedder.info().dimension).map_err(|e| ServiceError::Io(e.to_string()))?
        } else {
            HnswIndex::new(self.embedder.info().dimension, 10_000)
        };
        let mut vec_m: [&mut dyn chunking_store::VectorIndexMaintainer; 1] = [&mut hnsw];

        ingest_chunks_orchestrated(&mut repo, records, &text_m, &mut vec_m, vectors)
            .map_err(|e| ServiceError::Index(e.to_string()))?;

        // Persist HNSW snapshot if we touched vectors
        if vectors.is_some() {
            hnsw.save(&hdir).map_err(|e| ServiceError::Io(e.to_string()))?;
        }
        // Refresh resident cache
        if let Ok(mut guard) = self.hnsw.write() { *guard = Some(hnsw); }
        Ok(())
    }

    /// Ingest a file by path with progress/cancel support: chunk -> embed -> upsert -> index.
    pub fn ingest_file_with_progress(
        &self,
        path: &str,
        doc_id_hint: Option<&str>,
        cancel: Option<&CancelToken>,
        mut progress: Option<Box<dyn FnMut(ProgressEvent) + Send>>,
    ) -> Result<(), ServiceError> {
        let out = file_chunker::chunk_file_with_file_record(path);
        let mut records = out.chunks;

        // Stamp timestamps and optional doc_id override
        let now = Utc::now().to_rfc3339();
        for rec in &mut records {
            if let Some(h) = doc_id_hint { rec.doc_id = DocumentId(h.to_string()); }
            rec.extracted_at = now.clone();
        }

        if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::Start { total_chunks: records.len() }); }
        if let Some(ct) = cancel { if ct.is_canceled() { if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::Canceled); } return Err(ServiceError::Embed("canceled".into())); } }

        // Embed text (auto or fixed batches) to control memory
        let texts: Vec<&str> = records.iter().map(|c| c.text.as_str()).collect();
        let vecs = if self.cfg.embed_auto {
            let cb_opt: Option<&mut (dyn FnMut(ProgressEvent) + Send)> =
                progress.as_mut().map(|b| &mut **b as &mut (dyn FnMut(ProgressEvent) + Send));
            self.embed_texts_auto(&texts, cancel, cb_opt)?
        } else {
            let cb_opt: Option<&mut (dyn FnMut(ProgressEvent) + Send)> =
                progress.as_mut().map(|b| &mut **b as &mut (dyn FnMut(ProgressEvent) + Send));
            self.embed_texts_batched(&texts, cancel, cb_opt)?
        };
        if vecs.iter().any(|v| v.len() != self.embedder.info().dimension) {
            return Err(ServiceError::Embed("embedding dimension mismatch".into()));
        }
        let pairs: Vec<(ChunkId, Vec<f32>)> = records
            .iter()
            .zip(vecs.into_iter())
            .map(|(r, v)| (r.chunk_id.clone(), v))
            .collect();
        if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::UpsertDb { total: records.len() }); }
        self.ingest_chunks(&records, Some(&pairs))
            .and_then(|_| {
                if let Some(cb) = progress.as_deref_mut() {
                    cb(ProgressEvent::IndexText { total: records.len() });
                }
                if let Some(cb) = progress.as_deref_mut() {
                    cb(ProgressEvent::Finished { total: records.len() });
                }
                Ok(())
            })
    }

    /// Backwards compatible wrapper without progress/cancel.
    pub fn ingest_file(&self, path: &str, doc_id_hint: Option<&str>) -> Result<(), ServiceError> {
        self.ingest_file_with_progress(path, doc_id_hint, None, None)
    }

    /// Ingest a single text snippet as one chunk.
    pub fn ingest_text(&self, text: &str, doc_id_hint: Option<&str>) -> Result<(DocumentId, ChunkId), ServiceError> {
        let text = text.trim();
        if text.is_empty() { return Err(ServiceError::Embed("text is empty".into())); }
        // IDs
        let (doc_id, chunk_id) = make_ids_from_text(doc_id_hint, text);
        // Build record
        let rec = ChunkRecord {
            schema_version: chunk_model::SCHEMA_MAJOR,
            doc_id: doc_id.clone(),
            chunk_id: chunk_id.clone(),
            source_uri: "user://input".into(),
            source_mime: "text/plain".into(),
            extracted_at: Utc::now().to_rfc3339(),
            page_start: None,
            page_end: None,
            text: text.to_string(),
            section_path: None,
            meta: std::collections::BTreeMap::new(),
            extra: std::collections::BTreeMap::new(),
        };
        // Embed
        let vec = self.embedder.embed(text).map_err(|e| ServiceError::Embed(e.to_string()))?;
        // Upsert
        let vectors = vec![(rec.chunk_id.clone(), vec)];
        self.ingest_chunks(&[rec], Some(&vectors))?;
        Ok((doc_id, chunk_id))
    }

    /// Text-only search via FTS5 with filters.
    pub fn search_text(&self, query: &str, top_k: usize, filters: &[FilterClause]) -> Result<Vec<SearchHit>, ServiceError> {
        let repo = self.open_repo()?;
        let fts = Fts5Index::new();
        let opts = SearchOptions { top_k, fetch_factor: 10 };
        Ok(fts.search(&repo, query, filters, &opts))
    }

    /// Hybrid search: fuse FTS (text) and HNSW (vector) with weighted sum.
    pub fn search_hybrid(&self, query: &str, top_k: usize, filters: &[FilterClause], w_text: f32, w_vec: f32) -> Result<Vec<SearchHit>, ServiceError> {
        let repo = self.open_repo()?;
        let fts = Fts5Index::new();
        let opts = SearchOptions { top_k, fetch_factor: 10 };

        let mut text_matches = TextSearcher::search_ids(&fts, &repo, query, filters, &opts);

        // Vector query (ensure model warm)
        self.ensure_warm();
        let qvec = self.embedder.embed(query).map_err(|e| ServiceError::Embed(e.to_string()))?;
        // Use resident HNSW if available; lazily load once if missing
        let mut vec_matches: Vec<chunking_store::TextMatch> = Vec::new();
        let mut need_try_load = false;
        {
            if let Ok(guard) = self.hnsw.read() {
                if let Some(h) = guard.as_ref() {
                    vec_matches = VectorSearcher::knn_ids(h, &repo, &qvec, filters, &opts);
                } else {
                    need_try_load = true;
                }
            } else {
                need_try_load = true;
            }
        }
        if need_try_load {
            let hdir = self.hnsw_dir();
            if Path::new(&hdir).join("map.tsv").exists() {
                if let Ok(h) = HnswIndex::load(&hdir, qvec.len()) {
                    if let Ok(mut w) = self.hnsw.write() { *w = Some(h); }
                    if let Ok(r) = self.hnsw.read() {
                        if let Some(h) = r.as_ref() {
                            vec_matches = VectorSearcher::knn_ids(h, &repo, &qvec, filters, &opts);
                        }
                    }
                }
            }
        }

        // Combine
        let mut score_map: HashMap<String, f32> = HashMap::new();
        for m in text_matches.drain(..) {
            let e = score_map.entry(m.chunk_id.0).or_insert(0.0);
            *e += w_text * m.score;
        }
        for m in vec_matches.drain(..) {
            let e = score_map.entry(m.chunk_id.0).or_insert(0.0);
            *e += w_vec * m.score;
        }

        // Sort and materialize
        let mut items: Vec<(String, f32)> = score_map.into_iter().collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if items.len() > top_k { items.truncate(top_k); }

        let ids: Vec<ChunkId> = items.iter().map(|(cid, _)| ChunkId(cid.clone())).collect();
        let recs = repo.get_chunks_by_ids(&ids).map_err(|e| ServiceError::Repo(e.to_string()))?;
        // Build score map for quick lookup
        let mut cscore: HashMap<String, f32> = HashMap::new();
        for (cid, s) in items { cscore.insert(cid, s); }
        let mut out: Vec<SearchHit> = Vec::with_capacity(recs.len());
        for rec in recs {
            if let Some(score) = cscore.get(&rec.chunk_id.0) {
                out.push(SearchHit { chunk: rec, score: *score });
            }
        }
        Ok(out)
    }

    /// Delete by filters across DB and both indexes.
    pub fn delete_by_filter(&self, filters: &[FilterClause], batch_size: usize) -> Result<DeleteReport, ServiceError> {
        let mut repo = self.open_repo()?;
        let fts = Fts5Index::new();
        let text_m: [&dyn chunking_store::TextIndexMaintainer; 1] = [&fts];
        // Load HNSW (if exists)
        let hdir = self.hnsw_dir();
        let mut hnsw = if Path::new(&hdir).join("map.tsv").exists() {
            HnswIndex::load(&hdir, self.embedder.info().dimension).map_err(|e| ServiceError::Io(e.to_string()))?
        } else { HnswIndex::new(self.embedder.info().dimension, 10_000) };
        let mut vec_m: [&mut dyn chunking_store::VectorIndexMaintainer; 1] = [&mut hnsw];

        let rep = delete_by_filter_orchestrated(&mut repo, filters, batch_size, &text_m, &mut vec_m)
            .map_err(|e| ServiceError::Index(e.to_string()))?;

        // Persist HNSW snapshot post-delete and refresh resident cache
        hnsw.save(&hdir).map_err(|e| ServiceError::Io(e.to_string()))?;
        if let Ok(mut guard) = self.hnsw.write() { *guard = Some(hnsw); }
        Ok(rep)
    }

    /// Quick sanity/check API: counts for chunks and FTS mirror.
    pub fn repo_counts(&self) -> Result<(i64, i64), ServiceError> {
        let repo = self.open_repo()?;
        repo.counts().map_err(|e| ServiceError::Repo(e.to_string()))
    }

    /// Helper: embed texts in smaller batches according to config to limit memory spikes.
    fn embed_texts_batched<'p>(
        &self,
        texts: &[&str],
        cancel: Option<&CancelToken>,
        mut progress: Option<&'p mut (dyn FnMut(ProgressEvent) + Send)>,
    ) -> Result<Vec<Vec<f32>>, ServiceError> {
        if texts.is_empty() { return Ok(Vec::new()); }
        let bsz = self.cfg.embed_batch_size.max(1);
        let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        let mut done = 0usize;
        for chunk in texts.chunks(bsz) {
            if let Some(ct) = cancel { if ct.is_canceled() { if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::Canceled); } return Err(ServiceError::Embed("canceled".into())); } }
            let vecs = self
                .embedder
                .embed_batch(chunk)
                .map_err(|e| ServiceError::Embed(e.to_string()))?;
            out.extend(vecs);
            done += chunk.len();
            if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::EmbedBatch { done, total: texts.len(), batch: chunk.len() }); }
        }
        Ok(out)
    }

    /// Helper: auto batch sizing with simple bucketing by length and backoff on failure.
    fn embed_texts_auto<'p>(
        &self,
        texts: &[&str],
        cancel: Option<&CancelToken>,
        mut progress: Option<&'p mut (dyn FnMut(ProgressEvent) + Send)>,
    ) -> Result<Vec<Vec<f32>>, ServiceError> {
        if texts.is_empty() { return Ok(Vec::new()); }

        // Build (index, approx_len) and sort by length (ascending)
        let mut items: Vec<(usize, usize)> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.chars().count()))
            .collect();
        items.sort_by_key(|(_, l)| *l);

        let max_input = self.cfg.embedder.max_input_length.max(1);
        let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut i = 0;
        let mut done_total = 0usize;
        while i < items.len() {
            // Current bucket start
            let (_, l0) = items[i];
            // Bucket: group by similar length (within 1/4 max_input or same order of magnitude)
            let mut j = i;
            while j < items.len() {
                let (_, lj) = items[j];
                if (lj as isize - l0 as isize).unsigned_abs() > (max_input / 4) { break; }
                j += 1;
            }

            // Determine initial batch based on length factor
            let len_factor = (l0.min(max_input) as f32 / max_input as f32).max(0.1); // 0.1..1.0
            let mut bsz = ((self.cfg.embed_initial_batch as f32) * (1.0 / len_factor)).round() as usize;
            bsz = bsz.clamp(self.cfg.embed_min_batch, self.cfg.embed_initial_batch.max(self.cfg.embed_min_batch));

            let mut k = i;
            while k < j {
                let end = (k + bsz).min(j);
                let batch_idx: Vec<usize> = items[k..end].iter().map(|(idx, _)| *idx).collect();
                let batch_texts: Vec<&str> = batch_idx.iter().map(|&idx| texts[idx]).collect();

                if let Some(ct) = cancel { if ct.is_canceled() { if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::Canceled); } return Err(ServiceError::Embed("canceled".into())); } }

                match self.embedder.embed_batch(&batch_texts) {
                    Ok(vecs) => {
                        for (bi, v) in batch_idx.iter().zip(vecs.into_iter()) {
                            out[*bi] = Some(v);
                        }
                        k = end; // advance
                        done_total += batch_idx.len();
                        if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::EmbedBatch { done: done_total, total: texts.len(), batch: batch_idx.len() }); }
                        // Optional: slowly increase bsz on success
                        if bsz < self.cfg.embed_initial_batch { bsz = (bsz * 2).min(self.cfg.embed_initial_batch); }
                    }
                    Err(_e) => {
                        // Backoff and retry smaller batch
                        if bsz <= self.cfg.embed_min_batch { return Err(ServiceError::Embed("embed auto-batch failed even at minimum batch".into())); }
                        bsz = (bsz / 2).max(self.cfg.embed_min_batch);
                        continue;
                    }
                }
            }
            i = j;
        }

        // Collect preserving input order
        let mut result: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for v in out.into_iter() {
            match v { Some(vec) => result.push(vec), None => return Err(ServiceError::Embed("missing embedding output".into())) }
        }
        Ok(result)
    }
}

fn derive_hnsw_dir(db_path: &Path) -> PathBuf {
    let mut s = db_path.as_os_str().to_string_lossy().to_string();
    s.push_str(".hnsw");
    PathBuf::from(s)
}

fn make_ids_from_text(doc_hint: Option<&str>, text: &str) -> (DocumentId, ChunkId) {
    if let Some(h) = doc_hint { if !h.trim().is_empty() { return (DocumentId(h.to_string()), ChunkId(format!("{}#0", h))); } }
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    let h = hasher.finish();
    let ts = Utc::now().timestamp_millis();
    let doc_id = format!("doc-{ts:x}-{h:08x}");
    let chunk_id = format!("{}#0", doc_id);
    (DocumentId(doc_id), ChunkId(chunk_id))
}
