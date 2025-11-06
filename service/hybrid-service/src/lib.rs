use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, atomic::{AtomicBool, AtomicU64, Ordering}};

use chrono::Utc;
use chunk_model::{ChunkId, ChunkRecord, DocumentId, FileRecord};
#[cfg(all(not(feature = "tantivy"), feature = "fts"))]
use chunking_store::fts5_index::Fts5Index;
use chunking_store::hnsw_index::HnswIndex;
use chunking_store::orchestrator::{delete_by_filter_orchestrated, ingest_chunks_orchestrated, DeleteReport};
use chunking_store::{ChunkStoreRead, FilterClause, SearchHit, SearchOptions, VectorSearcher};
use chunking_store::sqlite_repo::SqliteRepo;
#[cfg(feature = "tantivy")]
use chunking_store::tantivy_index::{TantivyIndex, TokenCombine};
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
    /// When true, warm up HNSW + Tantivy + embedder concurrently.
    pub aggressive_warmup: bool,
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
            aggressive_warmup: true,
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
    // Active store paths (mutable at runtime)
    db_path: Arc<RwLock<PathBuf>>,
    hnsw_dir_override: Arc<RwLock<Option<PathBuf>>>,
    // Optional dynamic provider for store paths. When set, operations will
    // consult it and call set_store_paths if a change is detected.
    store_provider: Arc<RwLock<Option<Arc<dyn Fn() -> (PathBuf, Option<PathBuf>) + Send + Sync>>>>,
    // Resident index + state
    hnsw: Arc<RwLock<Option<HnswIndex>>>,
    warmed: AtomicBool,
    hnsw_state: Arc<RwLock<HnswState>>, 
    #[cfg(feature = "tantivy")]
    tantivy: Arc<RwLock<Option<TantivyIndex>>>,
    #[cfg(feature = "tantivy")]
    tantivy_state: Arc<RwLock<TantivyState>>,
    /// Monotonic epoch to invalidate stale background loads when paths change
    store_epoch: Arc<AtomicU64>,
}

/// State of the resident HNSW index in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HnswState { Absent, Loading, Ready, Error }

#[cfg(feature = "tantivy")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TantivyState { Absent, Loading, Ready, Error }

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
    /// Guarded access to the primary SQLite repo with store-path consistency.
    /// Ensures dynamic store paths are applied before opening and then
    /// provides `&SqliteRepo` to the caller-supplied function.
    pub fn with_repo<R, F>(&self, f: F) -> Result<R, ServiceError>
    where
        F: FnOnce(&SqliteRepo) -> Result<R, ServiceError>,
    {
        let repo = self.open_repo()?;
        f(&repo)
    }

    /// Guarded access to the resident HNSW index and repo with store-path consistency.
    /// If the HNSW snapshot exists at the current path but is not yet loaded,
    /// this attempts a lazy load. Returns Ok(None) when no index is available.
    pub fn with_hnsw<R, F>(&self, f: F) -> Result<Option<R>, ServiceError>
    where
        F: FnOnce(&HnswIndex, &SqliteRepo) -> R,
    {
        // Ensure we are pointing at up-to-date paths and open a repo
        self.ensure_store_paths_from_provider();
        let repo = self.open_repo()?;

        // Fast path: already loaded
        if let Ok(g) = self.hnsw.read() {
            if let Some(h) = g.as_ref() {
                return Ok(Some(f(h, &repo)));
            }
        }

        // Try lazy-load from current directory
        let hdir = self.hnsw_dir();
        let exists = std::path::Path::new(&hdir).join("map.tsv").exists();
        if exists {
            if let Ok(mut s) = self.hnsw_state.write() { *s = HnswState::Loading; }
            match HnswIndex::load(&hdir, self.embedder.info().dimension) {
                Ok(h) => {
                    let _ = self.hnsw.write().map(|mut w| *w = Some(h));
                    if let Ok(mut s) = self.hnsw_state.write() { *s = HnswState::Ready; }
                }
                Err(_) => { let _ = self.hnsw_state.write().map(|mut s| *s = HnswState::Error); }
            }
        } else {
            let _ = self.hnsw_state.write().map(|mut s| *s = HnswState::Absent);
        }

        // Re-check
        if let Ok(g) = self.hnsw.read() {
            if let Some(h) = g.as_ref() {
                return Ok(Some(f(h, &repo)));
            }
        }
        Ok(None)
    }
    pub fn new(cfg: ServiceConfig) -> Result<Self, ServiceError> {
        // Ensure DB dir exists
        if let Some(dir) = cfg.db_path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| ServiceError::Io(e.to_string()))?;
        }
        // Prepare runtime paths and resident index/state first so we can load HNSW in parallel with model init
        let db_path = Arc::new(RwLock::new(cfg.db_path.clone()));
        let hnsw_dir_override = Arc::new(RwLock::new(cfg.hnsw_dir.clone()));
        let hnsw = Arc::new(RwLock::new(None));
        let store_provider: Arc<RwLock<Option<Arc<dyn Fn() -> (PathBuf, Option<PathBuf>) + Send + Sync>>>> = Arc::new(RwLock::new(None));
        let hnsw_state = Arc::new(RwLock::new(HnswState::Absent));
        #[cfg(feature = "tantivy")]
        let tantivy: Arc<RwLock<Option<TantivyIndex>>> = Arc::new(RwLock::new(None));
        #[cfg(feature = "tantivy")]
        let tantivy_state: Arc<RwLock<TantivyState>> = Arc::new(RwLock::new(TantivyState::Absent));
        // Epoch counter to guard background loads from committing after store change
        let store_epoch: Arc<AtomicU64> = Arc::new(AtomicU64::new(1));

        // Derive HNSW dir from config and kick preload immediately using configured dimension
        let hdir = match &cfg.hnsw_dir { Some(d) => d.clone(), None => derive_hnsw_dir(&cfg.db_path) };
        let dim_cfg = cfg.embedder.dimension;
        {
            let cache = Arc::clone(&hnsw);
            let state = Arc::clone(&hnsw_state);
            let db_arc = Arc::clone(&db_path);
            let hnsw_arc = Arc::clone(&hnsw_dir_override);
            let dbp_for_warm = cfg.db_path.clone();
            let epoch_arc = Arc::clone(&store_epoch);
            let aggressive = cfg.aggressive_warmup;
            #[cfg(feature = "tantivy")]
            let tv_cache = Arc::clone(&tantivy);
            #[cfg(feature = "tantivy")]
            let tv_state = Arc::clone(&tantivy_state);
            std::thread::spawn(move || {
                // Verify target still current for this service instance (epoch + path)
                let epoch_start = epoch_arc.load(Ordering::SeqCst);
                let cur_db = db_arc.read().map(|p| p.clone()).unwrap_or_else(|_| dbp_for_warm.clone());
                let cur_h = hnsw_arc.read().ok().and_then(|g| g.clone());
                let cur_hdir = match cur_h { Some(d) => d, None => derive_hnsw_dir(&cur_db) };
                if cur_hdir != hdir {
                    // Store paths changed; abort without touching state/cache
                    return;
                }
                let exists = Path::new(&hdir).join("map.tsv").exists();
                if !exists {
                    if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                    let _ = state.write().map(|mut s| *s = HnswState::Absent);
                    return;
                }
                if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                let _ = state.write().map(|mut s| *s = HnswState::Loading);
                match HnswIndex::load(&hdir, dim_cfg) {
                    Ok(h) => {
                        // Re‑check on commit
                        let cur_db2 = db_arc.read().map(|p| p.clone()).unwrap_or_else(|_| dbp_for_warm.clone());
                        let cur_h2 = hnsw_arc.read().ok().and_then(|g| g.clone());
                        let cur_hdir2 = match cur_h2 { Some(d) => d, None => derive_hnsw_dir(&cur_db2) };
                        if cur_hdir2 != hdir || epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }

                        let _ = cache.write().map(|mut guard| *guard = Some(h));
                        // Optional KNN warm-up: open repo and run a trivial 1-NN to touch pages
                        if let Ok(repo) = SqliteRepo::open(&dbp_for_warm) {
                            let qvec = vec![0.0f32; dim_cfg];
                            let opts = SearchOptions { top_k: 1, fetch_factor: 1 };
                            if let Ok(guard) = cache.read() {
                                if let Some(h) = guard.as_ref() {
                                    let _ = VectorSearcher::knn_ids(h, &repo, &qvec, &[], &opts);
                                }
                            }
                        }
                        if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                        let _ = state.write().map(|mut s| *s = HnswState::Ready);
                    }
                    Err(_) => { if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; } let _ = state.write().map(|mut s| *s = HnswState::Error); }
                }

                // If aggressive OFF, sequentially open Tantivy after HNSW preload
                #[cfg(feature = "tantivy")]
                if !aggressive {
                    // Derive Tantivy dir from current DB path
                    let cur_db = db_arc.read().map(|p| p.clone()).unwrap_or_else(|_| dbp_for_warm.clone());
                    let tdir = cur_db.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(".")).join("tantivy");
                    if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                    let _ = tv_state.write().map(|mut s| *s = TantivyState::Loading);
                    if std::fs::create_dir_all(&tdir).is_err() {
                        if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                        let _ = tv_state.write().map(|mut s| *s = TantivyState::Error);
                        return;
                    }
                    match TantivyIndex::open_or_create_dir(&tdir) {
                        Ok(idx) => {
                            // Re-validate target before commit
                            let cur_db2 = db_arc.read().map(|p| p.clone()).unwrap_or_else(|_| dbp_for_warm.clone());
                            let tdir2 = cur_db2.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(".")).join("tantivy");
                            if tdir2 != tdir || epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                            let _ = tv_cache.write().map(|mut w| *w = Some(idx));
                            let _ = tv_state.write().map(|mut s| *s = TantivyState::Ready);
                        }
                        Err(_) => {
                            if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                            let _ = tv_state.write().map(|mut s| *s = TantivyState::Error);
                        }
                    }
                }
            });
        }

        // Background open/create Tantivy index at the store root (if enabled)
        #[cfg(feature = "tantivy")]
        if cfg.aggressive_warmup {
            // Derive Tantivy dir from DB path: <store_root>/tantivy
            let tdir = cfg
                .db_path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
                .join("tantivy");
            let tv_cache = Arc::clone(&tantivy);
            let tv_state = Arc::clone(&tantivy_state);
            let db_arc = Arc::clone(&db_path);
            let dbp_for_warm = cfg.db_path.clone();
            let epoch_arc = Arc::clone(&store_epoch);
            std::thread::spawn(move || {
                let epoch_start = epoch_arc.load(Ordering::SeqCst);
                // Verify that current derived tantivy dir still matches target (epoch + path)
                let cur_db = db_arc.read().map(|p| p.clone()).unwrap_or_else(|_| dbp_for_warm.clone());
                let cur_tdir = cur_db.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(".")).join("tantivy");
                if cur_tdir != tdir || epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                let _ = tv_state.write().map(|mut s| *s = TantivyState::Loading);
                if std::fs::create_dir_all(&tdir).is_err() {
                    if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                    let _ = tv_state.write().map(|mut s| *s = TantivyState::Error);
                    return;
                }
                match TantivyIndex::open_or_create_dir(&tdir) {
                    Ok(idx) => {
                        // Re-check before commit
                        let cur_db2 = db_arc.read().map(|p| p.clone()).unwrap_or_else(|_| dbp_for_warm.clone());
                        let cur_tdir2 = cur_db2.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(".")).join("tantivy");
                        if cur_tdir2 != tdir || epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                        let _ = tv_cache.write().map(|mut w| *w = Some(idx));
                        let _ = tv_state.write().map(|mut s| *s = TantivyState::Ready);
                    }
                    Err(_) => {
                        if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                        let _ = tv_state.write().map(|mut s| *s = TantivyState::Error);
                    }
                }
            });
        }

        // Initialize embedder (may run concurrently with HNSW loading)
        let embedder = OnnxStdIoEmbedder::new(cfg.embedder.clone())
            .map_err(|e| ServiceError::Embed(e.to_string()))?;

        let svc = Self {
            cfg,
            embedder,
            db_path,
            hnsw_dir_override,
            hnsw,
            warmed: AtomicBool::new(false),
            store_provider,
            hnsw_state,
            #[cfg(feature = "tantivy")]
            tantivy,
            #[cfg(feature = "tantivy")]
            tantivy_state,
            store_epoch,
        };
        // Warm up ONNX session once (best-effort) when aggressive
        if svc.cfg.aggressive_warmup {
            let _ = svc.embedder.embed("warmup").map(|_| svc.warmed.store(true, Ordering::Relaxed));
        }
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

    /// Current HNSW state (Absent/Loading/Ready/Error).
    pub fn hnsw_state(&self) -> HnswState {
        match self.hnsw_state.read() {
            Ok(s) => s.clone(),
            Err(_) => HnswState::Error,
        }
    }

    fn open_repo(&self) -> Result<SqliteRepo, ServiceError> {
        // Before opening, allow dynamic update of active paths.
        self.ensure_store_paths_from_provider();
        let path = self.db_path.read().map(|p| p.clone()).unwrap_or_else(|_| self.cfg.db_path.clone());
        let repo = SqliteRepo::open(&path).map_err(|e| ServiceError::Repo(e.to_string()))?;
        Ok(repo)
    }

    /// Install or replace the dynamic store path provider.
    pub fn set_store_path_provider(&self, provider: Arc<dyn Fn() -> (PathBuf, Option<PathBuf>) + Send + Sync>) {
        if let Ok(mut w) = self.store_provider.write() { *w = Some(provider); }
    }

    /// If a provider is installed, fetch current paths and apply when changed.
    fn ensure_store_paths_from_provider(&self) {
        let prov = match self.store_provider.read() { Ok(g) => g.clone(), Err(_) => None };
        if let Some(cb) = prov {
            let (new_db, new_hnsw) = cb();
            let cur_db = self.db_path.read().map(|p| p.clone()).unwrap_or_else(|_| self.cfg.db_path.clone());
            let cur_h = self.hnsw_dir_override.read().ok().and_then(|g| g.clone());
            let same_db = new_db == cur_db;
            let same_h = new_hnsw == cur_h;
            if !same_db || !same_h {
                self.set_store_paths(new_db, new_hnsw);
            }
        }
    }

    fn hnsw_dir(&self) -> PathBuf {
        // Prefer runtime override, otherwise derive from current db_path
        if let Ok(ovr) = self.hnsw_dir_override.read() {
            if let Some(d) = ovr.as_ref() { return d.clone(); }
        }
        let dbp = self.db_path.read().map(|p| p.clone()).unwrap_or_else(|_| self.cfg.db_path.clone());
        derive_hnsw_dir(&dbp)
    }

    #[cfg(feature = "tantivy")]
    fn tantivy_dir(&self) -> PathBuf {
        let dbp = self.db_path.read().map(|p| p.clone()).unwrap_or_else(|_| self.cfg.db_path.clone());
        let base = dbp.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."));
        base.join("tantivy")
    }

    #[cfg(feature = "tantivy")]
    pub fn with_tantivy<R, F>(&self, f: F) -> Result<Option<R>, ServiceError>
    where
        F: FnOnce(&TantivyIndex, &SqliteRepo) -> R,
    {
        self.ensure_store_paths_from_provider();
        let repo = self.open_repo()?;
        let need_open = match self.tantivy.read() { Ok(g) => g.is_none(), Err(_) => true };
        if need_open {
            let dir = self.tantivy_dir();
            std::fs::create_dir_all(&dir).map_err(|e| ServiceError::Io(e.to_string()))?;
            match TantivyIndex::open_or_create_dir(&dir) {
                Ok(idx) => {
                    let _ = self.tantivy.write().map(|mut w| *w = Some(idx));
                    let _ = self.tantivy_state.write().map(|mut s| *s = TantivyState::Ready);
                }
                Err(_) => {
                    let _ = self.tantivy_state.write().map(|mut s| *s = TantivyState::Error);
                    return Ok(None);
                }
            }
        }
        if let Ok(g) = self.tantivy.read() {
            if let Some(t) = g.as_ref() {
                return Ok(Some(f(t, &repo)));
            }
        }
        Ok(None)
    }

    #[cfg(feature = "tantivy")]
    pub fn tantivy_triple(&self, query: &str, top_k: usize, filters: &[FilterClause]) -> Result<(Vec<chunking_store::TextMatch>, Vec<chunking_store::TextMatch>, Vec<chunking_store::TextMatch>), ServiceError> {
        let opts = SearchOptions { top_k, fetch_factor: 10 };
        match self.with_tantivy(|ti, repo| {
            let a = chunking_store::TextSearcher::search_ids(ti, repo, query, filters, &opts);
            let b = ti.search_ids_tokenized(repo, query, filters, &opts, TokenCombine::AND);
            let c = ti.search_ids_tokenized(repo, query, filters, &opts, TokenCombine::OR);
            (a, b, c)
        })? {
            Some(x) => Ok(x),
            None => Ok((Vec::new(), Vec::new(), Vec::new())),
        }
    }

    /// Update active DB/HNSW paths at runtime and attempt to preload HNSW.
    pub fn set_store_paths(&self, db_path: PathBuf, hnsw_dir: Option<PathBuf>) {
        // Short-circuit when paths are unchanged to avoid resetting resident caches
        let cur_db = self
            .db_path
            .read()
            .map(|p| p.clone())
            .unwrap_or_else(|_| self.cfg.db_path.clone());
        let cur_h = self
            .hnsw_dir_override
            .read()
            .ok()
            .and_then(|g| g.clone());
        if cur_db == db_path && cur_h == hnsw_dir {
            // No-op: already pointing at requested paths
            return;
        }
        if let Ok(mut w) = self.db_path.write() { *w = db_path; }
        if let Ok(mut w) = self.hnsw_dir_override.write() { *w = hnsw_dir; }
        // Reset resident cache and state, and try loading if index exists
        let _ = self.hnsw.write().map(|mut w| *w = None);
        #[cfg(feature = "tantivy")]
        {
            let _ = self.tantivy.write().map(|mut w| *w = None);
            let _ = self.tantivy_state.write().map(|mut s| *s = TantivyState::Absent);
        }
        // Bump epoch to invalidate in-flight background loads for previous paths
        let _ = self.store_epoch.fetch_add(1, Ordering::SeqCst);
        let hdir = self.hnsw_dir();
        let dim = self.embedder.info().dimension;
        let db_for_warm = self.db_path.read().map(|p| p.clone()).unwrap_or_else(|_| self.cfg.db_path.clone());
        let cache = Arc::clone(&self.hnsw);
        let state = Arc::clone(&self.hnsw_state);
        let db_arc = Arc::clone(&self.db_path);
        let h_arc = Arc::clone(&self.hnsw_dir_override);
        let epoch_arc = Arc::clone(&self.store_epoch);
        let aggressive = self.cfg.aggressive_warmup;
        #[cfg(feature = "tantivy")]
        let tv_cache = Arc::clone(&self.tantivy);
        #[cfg(feature = "tantivy")]
        let tv_state = Arc::clone(&self.tantivy_state);
        std::thread::spawn(move || {
            let epoch_start = epoch_arc.load(Ordering::SeqCst);
            // Verify target still current
            let cur_db = db_arc.read().map(|p| p.clone()).unwrap_or_else(|_| db_for_warm.clone());
            let cur_h = h_arc.read().ok().and_then(|g| g.clone());
            let cur_hdir = match cur_h { Some(d) => d, None => derive_hnsw_dir(&cur_db) };
            if cur_hdir != hdir || epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
            let idx = Path::new(&hdir).join("map.tsv");
            if !idx.exists() {
                if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                let _ = state.write().map(|mut s| *s = HnswState::Absent);
                return;
            }
            if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
            let _ = state.write().map(|mut s| *s = HnswState::Loading);
            match HnswIndex::load(&hdir, dim) {
                Ok(h) => {
                    // Re-check before commit
                    let cur_db2 = db_arc.read().map(|p| p.clone()).unwrap_or_else(|_| db_for_warm.clone());
                    let cur_h2 = h_arc.read().ok().and_then(|g| g.clone());
                    let cur_hdir2 = match cur_h2 { Some(d) => d, None => derive_hnsw_dir(&cur_db2) };
                    if cur_hdir2 != hdir || epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                    let _ = cache.write().map(|mut w| *w = Some(h));
                    // KNN warm-up
                    if let Ok(repo) = SqliteRepo::open(&db_for_warm) {
                        let qvec = vec![0.0f32; dim];
                        let opts = SearchOptions { top_k: 1, fetch_factor: 1 };
                        if let Ok(guard) = cache.read() {
                            if let Some(h) = guard.as_ref() {
                                let _ = VectorSearcher::knn_ids(h, &repo, &qvec, &[], &opts);
                            }
                        }
                    }
                    if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                    let _ = state.write().map(|mut s| *s = HnswState::Ready);
                }
                Err(_) => { if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; } let _ = state.write().map(|mut s| *s = HnswState::Error); }
            }

            // If aggressive OFF, sequentially open Tantivy after HNSW reload
            #[cfg(feature = "tantivy")]
            if !aggressive {
                // Derive Tantivy dir from current DB path
                let cur_db = db_arc.read().map(|p| p.clone()).unwrap_or_else(|_| db_for_warm.clone());
                let tdir = cur_db.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(".")).join("tantivy");
                if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                let _ = tv_state.write().map(|mut s| *s = TantivyState::Loading);
                if std::fs::create_dir_all(&tdir).is_err() {
                    if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                    let _ = tv_state.write().map(|mut s| *s = TantivyState::Error);
                    return;
                }
                match TantivyIndex::open_or_create_dir(&tdir) {
                    Ok(idx) => {
                        // Re-check before commit
                        let cur_db2 = db_arc.read().map(|p| p.clone()).unwrap_or_else(|_| db_for_warm.clone());
                        let tdir2 = cur_db2.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(".")).join("tantivy");
                        if tdir2 != tdir || epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                        let _ = tv_cache.write().map(|mut w| *w = Some(idx));
                        let _ = tv_state.write().map(|mut s| *s = TantivyState::Ready);
                    }
                    Err(_) => {
                        if epoch_arc.load(Ordering::SeqCst) != epoch_start { return; }
                        let _ = tv_state.write().map(|mut s| *s = TantivyState::Error);
                    }
                }
            }
        });

        // Also (re)open Tantivy in background for the new store paths (if enabled)
        #[cfg(feature = "tantivy")]
        if self.cfg.aggressive_warmup {
            let tdir = self.tantivy_dir();
            let tv_cache = Arc::clone(&self.tantivy);
            let tv_state = Arc::clone(&self.tantivy_state);
            let db_arc2 = Arc::clone(&self.db_path);
            let epoch_arc2 = Arc::clone(&self.store_epoch);
            std::thread::spawn(move || {
                let epoch_start = epoch_arc2.load(Ordering::SeqCst);
                // Verify still current derived tantivy dir before state changes
                let cur_db = db_arc2.read().map(|p| p.clone()).unwrap_or_else(|_| PathBuf::from("."));
                let cur_tdir = cur_db.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(".")).join("tantivy");
                if cur_tdir != tdir || epoch_arc2.load(Ordering::SeqCst) != epoch_start { return; }
                let _ = tv_state.write().map(|mut s| *s = TantivyState::Loading);
                if std::fs::create_dir_all(&tdir).is_err() {
                    if epoch_arc2.load(Ordering::SeqCst) != epoch_start { return; }
                    let _ = tv_state.write().map(|mut s| *s = TantivyState::Error);
                    return;
                }
                match TantivyIndex::open_or_create_dir(&tdir) {
                    Ok(idx) => {
                        // Re-check before commit
                        let cur_db2 = db_arc2.read().map(|p| p.clone()).unwrap_or_else(|_| PathBuf::from("."));
                        let cur_tdir2 = cur_db2.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(".")).join("tantivy");
                        if cur_tdir2 != tdir || epoch_arc2.load(Ordering::SeqCst) != epoch_start { return; }
                        let _ = tv_cache.write().map(|mut w| *w = Some(idx));
                        let _ = tv_state.write().map(|mut s| *s = TantivyState::Ready);
                    }
                    Err(_) => {
                        if epoch_arc2.load(Ordering::SeqCst) != epoch_start { return; }
                        let _ = tv_state.write().map(|mut s| *s = TantivyState::Error);
                    }
                }
            });
        }
    }

    #[cfg(feature = "tantivy")]
    pub fn tantivy_state(&self) -> TantivyState {
        match self.tantivy_state.read() {
            Ok(s) => s.clone(),
            Err(_) => TantivyState::Error,
        }
    }

    /// Ingest pre-built chunks with optional precomputed vectors.
    pub fn ingest_chunks(&self, records: &[ChunkRecord], vectors: Option<&[(ChunkId, Vec<f32>)]>) -> Result<(), ServiceError> {
        if records.is_empty() { return Ok(()); }
        let mut repo = self.open_repo()?;

        // Prepare text index maintainers (optional FTS)
        #[cfg(feature = "fts")]
        let fts = chunking_store::fts5_index::Fts5Index::new();
        #[cfg(feature = "fts")]
        let text_m: [&dyn chunking_store::TextIndexMaintainer; 1] = [&fts];
        #[cfg(not(feature = "fts"))]
        let text_m: [&dyn chunking_store::TextIndexMaintainer; 0] = [];

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
        // Refresh resident cache and state
        if let Ok(mut guard) = self.hnsw.write() { *guard = Some(hnsw); }
        let _ = self.hnsw_state.write().map(|mut s| *s = HnswState::Ready);
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
        let mut file: FileRecord = out.file;
        let mut records = out.chunks;

        // Stamp timestamps and optional doc_id override
        let now = Utc::now().to_rfc3339();
        for rec in &mut records {
            if let Some(h) = doc_id_hint { rec.doc_id = DocumentId(h.to_string()); }
            rec.extracted_at = now.clone();
        }
        // FileRecord stamps
        if let Some(h) = doc_id_hint { file.doc_id = DocumentId(h.to_string()); }
        file.extracted_at = now.clone();
        file.chunk_count = Some(records.len() as u32);

        // Upsert FileRecord before chunk/vectors
        self.with_repo(|repo| repo.upsert_file(&file).map_err(|e| ServiceError::Repo(e.to_string())))?;

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
                #[cfg(feature = "tantivy")]
                { let _ = self.with_tantivy(|ti, _repo| { let _ = ti.upsert_records(&records); () }); }
                if let Some(cb) = progress.as_deref_mut() {
                    cb(ProgressEvent::IndexText { total: records.len() });
                }
                if let Some(cb) = progress.as_deref_mut() {
                    cb(ProgressEvent::Finished { total: records.len() });
                }
                Ok(())
            })
    }

    /// Variant of ingest_file_with_progress that allows specifying text encoding for text-like files.
    pub fn ingest_file_with_progress_with_encoding(
        &self,
        path: &str,
        doc_id_hint: Option<&str>,
        encoding: Option<&str>,
        cancel: Option<&CancelToken>,
        mut progress: Option<Box<dyn FnMut(ProgressEvent) + Send>>,
    ) -> Result<(), ServiceError> {
        // Use encoding-aware path for text-like files; for others it's identical
        let out = file_chunker::chunk_file_with_file_record_with_encoding(path, encoding);
        let mut file: FileRecord = out.file;
        let mut records = out.chunks;

        // Stamp timestamps and optional doc_id override
        let now = Utc::now().to_rfc3339();
        for rec in &mut records {
            if let Some(h) = doc_id_hint { rec.doc_id = DocumentId(h.to_string()); }
            rec.extracted_at = now.clone();
        }
        if let Some(h) = doc_id_hint { file.doc_id = DocumentId(h.to_string()); }
        file.extracted_at = now.clone();
        file.chunk_count = Some(records.len() as u32);

        self.with_repo(|repo| repo.upsert_file(&file).map_err(|e| ServiceError::Repo(e.to_string())))?;

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
                #[cfg(feature = "tantivy")]
                { let _ = self.with_tantivy(|ti, _repo| { let _ = ti.upsert_records(&records); () }); }
                if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::IndexText { total: records.len() }); }
                if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::Finished { total: records.len() }); }
                Ok(())
            })
    }

    /// Ingest with explicit chunking parameters (min/max/cap and penalties) and optional encoding for text-like files.
    pub fn ingest_file_with_progress_custom(
        &self,
        path: &str,
        doc_id_hint: Option<&str>,
        encoding: Option<&str>,
        min_chars: usize,
        max_chars: usize,
        cap_chars: usize,
        short_merge_min_chars: usize,
        penalize_short_line: bool,
        penalize_page_boundary_no_newline: bool,
        cancel: Option<&CancelToken>,
        mut progress: Option<Box<dyn FnMut(ProgressEvent) + Send>>,
    ) -> Result<(), ServiceError> {
        let tparams = file_chunker::text_segmenter::TextChunkParams {
            min_chars,
            max_chars,
            cap_chars,
            penalize_short_line,
            penalize_page_boundary_no_newline,
            short_merge_min_chars,
        };
        let out = file_chunker::chunk_file_with_file_record_with_params(path, encoding, &tparams);
        let mut file: FileRecord = out.file;
        let mut records = out.chunks;

        // Stamp timestamps and optional doc_id override
        let now = Utc::now().to_rfc3339();
        for rec in &mut records {
            if let Some(h) = doc_id_hint { rec.doc_id = DocumentId(h.to_string()); }
            rec.extracted_at = now.clone();
        }
        if let Some(h) = doc_id_hint { file.doc_id = DocumentId(h.to_string()); }
        file.extracted_at = now.clone();
        file.chunk_count = Some(records.len() as u32);

        self.with_repo(|repo| repo.upsert_file(&file).map_err(|e| ServiceError::Repo(e.to_string())))?;


        if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::Start { total_chunks: records.len() }); }
        if let Some(ct) = cancel { if ct.is_canceled() { if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::Canceled); } return Err(ServiceError::Embed("canceled".into())); } }

        // Embed
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
                #[cfg(feature = "tantivy")]
                { let _ = self.with_tantivy(|ti, _repo| { let _ = ti.upsert_records(&records); () }); }
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
        #[cfg(feature = "tantivy")]
        {
            // Upsert just-inserted record into Tantivy at current path
            let records: Vec<ChunkRecord> = {
                let mut r = Vec::new();
                // recreate minimal record for upsert
                r.push(ChunkRecord {
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
                });
                r
            };
            let _ = self.with_tantivy(|ti, _repo| { let _ = ti.upsert_records(&records); () });
        }
        Ok((doc_id, chunk_id))
    }

    /// Ingest a single text snippet with progress callbacks and optional cancel.
    pub fn ingest_text_with_progress(
        &self,
        text: &str,
        doc_id_hint: Option<&str>,
        cancel: Option<&CancelToken>,
        mut progress: Option<Box<dyn FnMut(ProgressEvent) + Send>>,
    ) -> Result<(DocumentId, ChunkId), ServiceError> {
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
        // Progress start (total chunks = 1)
        if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::Start { total_chunks: 1 }); }
        if let Some(ct) = cancel { if ct.is_canceled() { if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::Canceled); } return Err(ServiceError::Embed("canceled".into())); } }
        // Embed via existing batched helper (one item)
        let texts_refs: [&str; 1] = [text];
        let vecs = {
            let cb_opt: Option<&mut (dyn FnMut(ProgressEvent) + Send)> =
                progress.as_mut().map(|b| &mut **b as &mut (dyn FnMut(ProgressEvent) + Send));
            self.embed_texts_batched(&texts_refs, cancel, cb_opt)?
        };
        // Upsert
        if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::UpsertDb { total: 1 }); }
        let vectors = vec![(rec.chunk_id.clone(), vecs.into_iter().next().unwrap_or_default())];
        self.ingest_chunks(&[rec], Some(&vectors))?;
        #[cfg(feature = "tantivy")]
        {
            let records: Vec<ChunkRecord> = {
                let mut r = Vec::new();
                r.push(ChunkRecord {
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
                });
                r
            };
            let _ = self.with_tantivy(|ti, _repo| { let _ = ti.upsert_records(&records); () });
        }
        if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::IndexText { total: 1 }); }
        if let Some(cb) = progress.as_deref_mut() { cb(ProgressEvent::Finished { total: 1 }); }
        Ok((doc_id, chunk_id))
    }

    /// Text-only search (prefer Tantivy when available) with filters.
    #[cfg(feature = "tantivy")]
    pub fn search_text(&self, query: &str, top_k: usize, filters: &[FilterClause]) -> Result<Vec<SearchHit>, ServiceError> {
        let opts = SearchOptions { top_k, fetch_factor: 10 };
        let tmatches: Vec<chunking_store::TextMatch> = match self.with_tantivy(|ti, repo| chunking_store::TextSearcher::search_ids(ti, repo, query, filters, &opts))? {
            Some(v) => v,
            None => Vec::new(),
        };
        if tmatches.is_empty() { return Ok(Vec::new()); }
        // Map scores by chunk_id for materialization
        let mut score_map: HashMap<String, f32> = HashMap::new();
        let ids: Vec<ChunkId> = tmatches
            .into_iter()
            .map(|m| { score_map.insert(m.chunk_id.0.clone(), m.score); m.chunk_id })
            .collect();
        let recs = self.with_repo(|repo| repo.get_chunks_by_ids(&ids).map_err(|e| ServiceError::Repo(e.to_string())))?;
        let mut out: Vec<SearchHit> = Vec::with_capacity(recs.len());
        for rec in recs {
            if let Some(score) = score_map.get(&rec.chunk_id.0) {
                out.push(SearchHit { chunk: rec, score: *score });
            }
        }
        Ok(out)
    }

    /// Fallback text-only search via FTS5 when Tantivy feature is disabled.
    #[cfg(all(not(feature = "tantivy"), feature = "fts"))]
    pub fn search_text(&self, query: &str, top_k: usize, filters: &[FilterClause]) -> Result<Vec<SearchHit>, ServiceError> {
        let fts = chunking_store::fts5_index::Fts5Index::new();
        let opts = SearchOptions { top_k, fetch_factor: 10 };
        self.with_repo(|repo| Ok(fts.search(repo, query, filters, &opts)))
    }
    /// Fallback when neither Tantivy nor FTS are enabled: return empty.
    #[cfg(all(not(feature = "tantivy"), not(feature = "fts")))]
    pub fn search_text(&self, _query: &str, _top_k: usize, _filters: &[FilterClause]) -> Result<Vec<SearchHit>, ServiceError> {
        Ok(Vec::new())
    }

    /// Hybrid search: fuse Text (Tantivy or FTS) and HNSW (vector) with weighted sum.
    pub fn search_hybrid(&self, query: &str, top_k: usize, filters: &[FilterClause], w_text: f32, w_vec: f32) -> Result<Vec<SearchHit>, ServiceError> {
        let opts = SearchOptions { top_k, fetch_factor: 10 };

        // Text matches (prefer Tantivy when enabled)
        #[cfg(feature = "tantivy")]
        let mut text_matches: Vec<chunking_store::TextMatch> = match self.with_tantivy(|ti, repo| chunking_store::TextSearcher::search_ids(ti, repo, query, filters, &opts))? {
            Some(v) => v,
            None => Vec::new(),
        };
        #[cfg(all(not(feature = "tantivy"), feature = "fts"))]
        let mut text_matches: Vec<chunking_store::TextMatch> = {
            let fts = chunking_store::fts5_index::Fts5Index::new();
            self.with_repo(|repo| Ok(chunking_store::TextSearcher::search_ids(&fts, repo, query, filters, &opts)))?
        };
        #[cfg(all(not(feature = "tantivy"), not(feature = "fts")))]
        let mut text_matches: Vec<chunking_store::TextMatch> = Vec::new();

        // Vector matches via HNSW guard (optional)
        self.ensure_warm();
        let qvec = self.embedder.embed(query).map_err(|e| ServiceError::Embed(e.to_string()))?;
        let vec_matches: Vec<chunking_store::TextMatch> = match self.with_hnsw(|h, repo| VectorSearcher::knn_ids(h, repo, &qvec, filters, &opts))? {
            Some(v) => v,
            None => Vec::new(),
        };

        // Combine scores
        let mut score_map: HashMap<String, f32> = HashMap::new();
        for m in text_matches.drain(..) {
            let e = score_map.entry(m.chunk_id.0).or_insert(0.0);
            *e += w_text * m.score;
        }
        for m in vec_matches.into_iter() {
            let e = score_map.entry(m.chunk_id.0).or_insert(0.0);
            *e += w_vec * m.score;
        }

        // Rank and materialize
        let mut items: Vec<(String, f32)> = score_map.into_iter().collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if items.len() > top_k { items.truncate(top_k); }

        let ids: Vec<ChunkId> = items.iter().map(|(cid, _)| ChunkId(cid.clone())).collect();
        let recs = self.with_repo(|repo| repo.get_chunks_by_ids(&ids).map_err(|e| ServiceError::Repo(e.to_string())))?;
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
        #[cfg(feature = "fts")]
        let fts = chunking_store::fts5_index::Fts5Index::new();
        #[cfg(feature = "fts")]
        let text_m: [&dyn chunking_store::TextIndexMaintainer; 1] = [&fts];
        #[cfg(not(feature = "fts"))]
        let text_m: [&dyn chunking_store::TextIndexMaintainer; 0] = [];
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
        let _ = self.hnsw_state.write().map(|mut s| *s = HnswState::Ready);

        // Keep files table in sync: targeted delete for DocId filters, then orphan cleanup
        // Targeted deletes
        let mut doc_ids: Vec<String> = Vec::new();
        for f in filters {
            match &f.op {
                chunking_store::FilterOp::DocIdEq(v) => doc_ids.push(v.clone()),
                chunking_store::FilterOp::DocIdIn(vs) => { for v in vs { doc_ids.push(v.clone()); } },
                _ => {}
            }
        }
        if !doc_ids.is_empty() {
            let _ = repo.delete_files_by_doc_ids(&doc_ids);
        }
        // Best-effort orphan cleanup (ignore errors)
        let _ = repo.cleanup_orphan_files();
        Ok(rep)
    }

    /// Quick sanity/check API: counts for chunks and FTS mirror.
    pub fn repo_counts(&self) -> Result<(i64, i64), ServiceError> {
        let repo = self.open_repo()?;
        repo.counts().map_err(|e| ServiceError::Repo(e.to_string()))
    }

    /// Return previous and next chunks within the same document for navigation.
    pub fn neighbor_chunks(&self, chunk_id: &str) -> Result<(Option<ChunkRecord>, Option<ChunkRecord>), ServiceError> {
        self.with_repo(|repo| repo
            .get_neighbor_chunks(&ChunkId(chunk_id.to_string()))
            .map_err(|e| ServiceError::Repo(e.to_string())))
    }

    /// List FileRecords with pagination (for GUI file list).
    pub fn list_files(&self, limit: usize, offset: usize) -> Result<Vec<FileRecord>, ServiceError> {
        self.with_repo(|repo| repo.list_files(limit, offset).map_err(|e| ServiceError::Repo(e.to_string())))
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
