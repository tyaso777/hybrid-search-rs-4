use std::path::{Path, PathBuf};
use std::collections::HashMap;

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
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            db_path: PathBuf::from("target/demo/chunks.db"),
            hnsw_dir: None,
            embedder: default_stdio_config(),
        }
    }
}

pub struct HybridService {
    cfg: ServiceConfig,
    embedder: OnnxStdIoEmbedder,
}

impl HybridService {
    pub fn new(cfg: ServiceConfig) -> Result<Self, ServiceError> {
        // Ensure DB dir exists
        if let Some(dir) = cfg.db_path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| ServiceError::Io(e.to_string()))?;
        }
        let embedder = OnnxStdIoEmbedder::new(cfg.embedder.clone())
            .map_err(|e| ServiceError::Embed(e.to_string()))?;
        Ok(Self { cfg, embedder })
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
        Ok(())
    }

    /// Ingest a file by path: chunk -> embed -> upsert -> index.
    pub fn ingest_file(&self, path: &str, doc_id_hint: Option<&str>) -> Result<(), ServiceError> {
        let out = file_chunker::chunk_file_with_file_record(path);
        let mut records = out.chunks;

        // Stamp timestamps and optional doc_id override
        let now = Utc::now().to_rfc3339();
        for rec in &mut records {
            if let Some(h) = doc_id_hint { rec.doc_id = DocumentId(h.to_string()); }
            rec.extracted_at = now.clone();
        }

        // Embed text in batch
        let texts: Vec<&str> = records.iter().map(|c| c.text.as_str()).collect();
        let vecs = self.embedder
            .embed_batch(&texts)
            .map_err(|e| ServiceError::Embed(e.to_string()))?;
        if vecs.iter().any(|v| v.len() != self.embedder.info().dimension) {
            return Err(ServiceError::Embed("embedding dimension mismatch".into()));
        }
        let pairs: Vec<(ChunkId, Vec<f32>)> = records
            .iter()
            .zip(vecs.into_iter())
            .map(|(r, v)| (r.chunk_id.clone(), v))
            .collect();

        self.ingest_chunks(&records, Some(&pairs))
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

        // Vector query
        let qvec = self.embedder.embed(query).map_err(|e| ServiceError::Embed(e.to_string()))?;
        let hdir = self.hnsw_dir();
        let maybe_hnsw = if Path::new(&hdir).join("map.tsv").exists() {
            match HnswIndex::load(&hdir, qvec.len()) { Ok(h) => Some(h), Err(_) => None }
        } else { None };
        let mut vec_matches = if let Some(h) = &maybe_hnsw {
            VectorSearcher::knn_ids(h, &repo, &qvec, filters, &opts)
        } else { Vec::new() };

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

        // Persist HNSW snapshot post-delete
        hnsw.save(&hdir).map_err(|e| ServiceError::Io(e.to_string()))?;
        Ok(rep)
    }

    /// Quick sanity/check API: counts for chunks and FTS mirror.
    pub fn repo_counts(&self) -> Result<(i64, i64), ServiceError> {
        let repo = self.open_repo()?;
        repo.counts().map_err(|e| ServiceError::Repo(e.to_string()))
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
