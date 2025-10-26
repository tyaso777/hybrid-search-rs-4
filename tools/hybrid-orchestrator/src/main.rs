use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use chunk_model::{ChunkId, ChunkRecord, DocumentId, SCHEMA_MAJOR};
use chunking_store::fts5_index::Fts5Index;
use chunking_store::hnsw_index::HnswIndex;
use chunking_store::orchestrator::ingest_chunks_orchestrated;
use chunking_store::{SearchOptions, TextSearcher, VectorSearcher, ChunkStoreRead};
use chunking_store::sqlite_repo::SqliteRepo;
use embedding_provider::config::default_stdio_config;
use embedding_provider::embedder::{Embedder, OnnxStdIoConfig, OnnxStdIoEmbedder};

fn print_usage() {
    eprintln!(
        "Usage:\n\
         hybrid-orchestrator insert [db_path] --text TEXT [--doc DOC] [--hnsw DIR]\n\
         hybrid-orchestrator insert [db_path] --stdin [--doc DOC] [--hnsw DIR]\n\
         hybrid-orchestrator search [db_path] --query Q [--k N] [--hybrid] [--hnsw DIR]\n\
         \n\
         Model overrides (optional for insert/search --hybrid):\n\
           --model PATH_ONNX   --tokenizer PATH_JSON   --runtime PATH_DLL   --dim N   --max-tokens N\n\
         Notes: db_path defaults to target/demo/chunks.db; hnsw defaults to <db_path>.hnsw\n"
    );
}

fn ensure_parent_dir(path: &str) -> std::io::Result<()> {
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn derive_hnsw_dir(db_path: &str) -> String {
    format!("{}.hnsw", db_path)
}

fn now_iso() -> String { chrono::Utc::now().to_rfc3339() }

fn make_ids_from_text(doc_hint: Option<String>, text: &str) -> (DocumentId, ChunkId) {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    let h = hasher.finish();
    let ts = chrono::Utc::now().timestamp_millis();
    let doc_id = if let Some(d) = doc_hint { d } else { format!("doc-{ts:x}-{h:08x}") };
    let chunk_id = format!("{}#0", doc_id);
    (DocumentId(doc_id), ChunkId(chunk_id))
}

fn build_embedder_from_args(args: &[String]) -> Result<OnnxStdIoEmbedder, String> {
    // Defaults from embedding_provider
    let mut cfg: OnnxStdIoConfig = default_stdio_config();

    // Parse overrides
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { if i+1<args.len() { cfg.model_path = PathBuf::from(&args[i+1]); i+=2; } else { return Err("--model requires path".into()); } }
            "--tokenizer" => { if i+1<args.len() { cfg.tokenizer_path = PathBuf::from(&args[i+1]); i+=2; } else { return Err("--tokenizer requires path".into()); } }
            "--runtime" => { if i+1<args.len() { cfg.runtime_library_path = PathBuf::from(&args[i+1]); i+=2; } else { return Err("--runtime requires path".into()); } }
            "--dim" => { if i+1<args.len() { cfg.dimension = args[i+1].parse().unwrap_or(cfg.dimension); i+=2; } else { return Err("--dim requires number".into()); } }
            "--max-tokens" => { if i+1<args.len() { cfg.max_input_length = args[i+1].parse().unwrap_or(cfg.max_input_length); i+=2; } else { return Err("--max-tokens requires number".into()); } }
            _ => { i+=1; }
        }
    }

    OnnxStdIoEmbedder::new(cfg).map_err(|e| format!("embedder init failed: {e}"))
}

fn do_insert(mut tail: Vec<String>) -> Result<(), String> {
    let default_db = String::from("target/demo/chunks.db");
    // Detect optional db path as first positional arg
    let (db_path, mut rest): (String, Vec<String>) = if !tail.is_empty() && !tail[0].starts_with('-') {
        (tail.remove(0), tail)
    } else {
        (default_db, tail)
    };

    let mut text: Option<String> = None;
    let mut use_stdin = false;
    let mut doc_hint: Option<String> = None;
    let mut hnsw_dir: Option<String> = None;

    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--text" => { if i+1<rest.len() { text = Some(rest[i+1].clone()); i+=2; } else { return Err("--text requires value".into()); } }
            "--stdin" => { use_stdin = true; i+=1; }
            "--doc" => { if i+1<rest.len() { doc_hint = Some(rest[i+1].clone()); i+=2; } else { return Err("--doc requires value".into()); } }
            "--hnsw" => { if i+1<rest.len() { hnsw_dir = Some(rest[i+1].clone()); i+=2; } else { return Err("--hnsw requires dir".into()); } }
            _ => { i+=1; }
        }
    }

    let input_text = if let Some(t) = text { t } else if use_stdin { 
        use std::io::Read; let mut buf = String::new(); std::io::stdin().read_to_string(&mut buf).map_err(|e| e.to_string())?; buf
    } else {
        return Err("provide --text or --stdin".into());
    };

    ensure_parent_dir(&db_path).map_err(|e| e.to_string())?;
    let mut repo = SqliteRepo::open(&db_path).map_err(|e| e.to_string())?;
    let _ = repo.maybe_rebuild_fts();

    let embedder = build_embedder_from_args(&rest)?;
    let vector = embedder.embed(&input_text).map_err(|e| format!("embed failed: {e}"))?;

    // Build record
    let (doc_id, chunk_id) = make_ids_from_text(doc_hint, &input_text);
    let mut meta: BTreeMap<String, String> = BTreeMap::new();
    meta.insert("ingest.ts".into(), now_iso());
    meta.insert("len".into(), input_text.len().to_string());
    let rec = ChunkRecord {
        schema_version: SCHEMA_MAJOR,
        doc_id: doc_id.clone(),
        chunk_id: chunk_id.clone(),
        source_uri: "user://input".into(),
        source_mime: "text/plain".into(),
        extracted_at: now_iso(),
        text: input_text.clone(),
        section_path: vec![],
        meta,
        extra: BTreeMap::new(),
    };

    // Prepare indexes
    let fts = Fts5Index::new();
    let text_indexes: [&dyn chunking_store::TextIndexMaintainer; 1] = [&fts];

    // HNSW vector index: load or init, then upsert, then save snapshot
    let hdir = hnsw_dir.unwrap_or_else(|| derive_hnsw_dir(&db_path));
    let mut hnsw = if Path::new(&hdir).join("map.tsv").exists() {
        HnswIndex::load(&hdir, vector.len()).map_err(|e| format!("load HNSW: {e}"))?
    } else {
        HnswIndex::new(vector.len(), 10_000)
    };
    let mut vec_indexes: [&mut dyn chunking_store::VectorIndexMaintainer; 1] = [&mut hnsw];
    let vectors: Vec<(ChunkId, Vec<f32>)> = vec![(chunk_id.clone(), vector.clone())];

    ingest_chunks_orchestrated(&mut repo, &[rec], &text_indexes, &mut vec_indexes, Some(&vectors))
        .map_err(|e| format!("ingest orchestrator failed: {e}"))?;

    // Persist HNSW snapshot
    hnsw.save(&hdir).map_err(|e| format!("save HNSW: {e}"))?;

    println!("Inserted chunk: {} (doc={})", chunk_id.0, doc_id.0);
    Ok(())
}

fn do_search(mut tail: Vec<String>) -> Result<(), String> {
    let default_db = String::from("target/demo/chunks.db");
    let (db_path, mut rest): (String, Vec<String>) = if !tail.is_empty() && !tail[0].starts_with('-') {
        (tail.remove(0), tail)
    } else {
        (default_db, tail)
    };

    let mut query: Option<String> = None;
    let mut k: usize = 10;
    let mut do_hybrid = false;
    let mut hnsw_dir: Option<String> = None;

    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--query" => { if i+1<rest.len() { query = Some(rest[i+1].clone()); i+=2; } else { return Err("--query requires value".into()); } }
            "--k" => { if i+1<rest.len() { k = rest[i+1].parse().unwrap_or(10); i+=2; } else { return Err("--k requires number".into()); } }
            "--hybrid" => { do_hybrid = true; i+=1; }
            "--hnsw" => { if i+1<rest.len() { hnsw_dir = Some(rest[i+1].clone()); i+=2; } else { return Err("--hnsw requires dir".into()); } }
            _ => { i+=1; }
        }
    }

    let q = match query { Some(s) => s, None => return Err("--query required".into()) };

    let repo = SqliteRepo::open(&db_path).map_err(|e| e.to_string())?;
    let _ = repo.maybe_rebuild_fts();
    let fts = Fts5Index::new();
    let opts = SearchOptions { top_k: k, fetch_factor: 10 };

    // Text-only path
    if !do_hybrid {
        let hits = fts.search(&repo, &q, &[], &opts);
        println!("FTS hits: {}", hits.len());
        for (i, h) in hits.iter().enumerate() {
            let text = &h.chunk.text;
            let preview = truncate_chars(text, 60);
            println!("{:>2}. [{}] score={:.4} {}", i+1, h.chunk.chunk_id.0, h.score, preview);
        }
        return Ok(());
    }

    // Hybrid: FTS + vector
    let embedder = build_embedder_from_args(&rest)?;
    let qvec = embedder.embed(&q).map_err(|e| format!("embed failed: {e}"))?;

    // Load HNSW (optional)
    let hdir = hnsw_dir.unwrap_or_else(|| derive_hnsw_dir(&db_path));
    let maybe_hnsw = if Path::new(&hdir).join("map.tsv").exists() {
        match HnswIndex::load(&hdir, qvec.len()) { Ok(h) => Some(h), Err(e) => { eprintln!("warn: HNSW load: {}", e); None } }
    } else { None };

    let mut text_matches = TextSearcher::search_ids(&fts, &repo, &q, &[], &opts);
    let mut vec_matches = if let Some(h) = &maybe_hnsw {
        VectorSearcher::knn_ids(h, &repo, &qvec, &[], &opts)
    } else { Vec::new() };

    // Combine with simple weighted sum (0.5 / 0.5)
    let w_text = 0.5f32;
    let w_vec = 0.5f32;

    let mut score_map: HashMap<String, f32> = HashMap::new();
    for m in text_matches.drain(..) {
        let e = score_map.entry(m.chunk_id.0).or_insert(0.0);
        *e += w_text * m.score;
    }
    for m in vec_matches.drain(..) {
        let e = score_map.entry(m.chunk_id.0).or_insert(0.0);
        *e += w_vec * m.score;
    }

    // Sort by combined score
    let mut items: Vec<(String, f32)> = score_map.into_iter().collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if items.len() > k { items.truncate(k); }

    let ids: Vec<ChunkId> = items.iter().map(|(cid, _)| ChunkId(cid.clone())).collect();
    let recs = repo.get_chunks_by_ids(&ids).map_err(|e| e.to_string())?;

    // Build a quick index to map chunk_id -> combined score
    let mut cscore: HashMap<String, f32> = HashMap::new();
    for (cid, s) in items { cscore.insert(cid, s); }

    println!("Hybrid hits: {}", recs.len());
    for (i, rec) in recs.iter().enumerate() {
        let score = cscore.get(&rec.chunk_id.0).copied().unwrap_or(0.0);
        let text = &rec.text;
        let preview = truncate_chars(text, 60);
        println!("{:>2}. [{}] score={:.4} {}", i+1, rec.chunk_id.0, score, preview);
    }

    Ok(())
}

fn main() {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() { print_usage(); return; }
    let cmd = args.remove(0);
    let res = match cmd.as_str() {
        "insert" => do_insert(args),
        "search" => do_search(args),
        _ => { print_usage(); return; }
    };
    if let Err(err) = res {
        eprintln!("Error: {}", err);
        print_usage();
    }
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if max_chars == 0 { return String::new(); }
    let mut it = s.chars();
    let truncated: String = it.by_ref().take(max_chars).collect();
    if it.next().is_some() { format!("{}…", truncated) } else { truncated }
}
