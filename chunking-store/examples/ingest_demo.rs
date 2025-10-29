use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

use chunk_model::{ChunkRecord, DocumentId, ChunkId, SCHEMA_MAJOR};
use chunking_store::fts5_index::Fts5Index;
use chunking_store::orchestrator::ingest_chunks_orchestrated;
use chunking_store::sqlite_repo::SqliteRepo;

fn print_usage() {
    eprintln!(
        "Usage: ingest_demo [db_path] [--ndjson PATH | --sample] [--search QUERY] [--debug]\n\
         Examples:\n\
           ingest_demo                --sample --search こんにちは   (uses target/demo/chunks.db)\n\
           ingest_demo ./chunks.db    --ndjson ./chunks.ndjson --search hello\n"
    );
}

fn load_ndjson(path: &str) -> Result<Vec<ChunkRecord>, Box<dyn std::error::Error>> {
    let f = File::open(path)?;
    let r = BufReader::new(f);
    let mut out = Vec::new();
    for line in r.lines() {
        let l = line?;
        if l.trim().is_empty() { continue; }
        let rec: ChunkRecord = serde_json::from_str(&l)?;
        rec.validate_soft().map_err(|e| format!("{e}"))?;
        out.push(rec);
    }
    Ok(out)
}

fn make_sample() -> Vec<ChunkRecord> {
    vec![
        ChunkRecord {
            schema_version: SCHEMA_MAJOR,
            doc_id: DocumentId("doc-001".into()),
            chunk_id: ChunkId("doc-001#0".into()),
            source_uri: "file:///sample/ja.txt".into(),
            source_mime: "text/plain".into(),
            extracted_at: "2024-06-01T00:00:00Z".into(),
            text: "こんにちは 世界。日本語の分かち書きテスト。".into(),
            section_path: Some(vec!["はじめに".into()]),
            meta: Default::default(),
            extra: Default::default(),
        },
        ChunkRecord {
            schema_version: SCHEMA_MAJOR,
            doc_id: DocumentId("doc-002".into()),
            chunk_id: ChunkId("doc-002#0".into()),
            source_uri: "file:///sample/en.txt".into(),
            source_mime: "text/plain".into(),
            extracted_at: "2024-07-01T00:00:00Z".into(),
            text: "hello world. this is a sample English chunk.".into(),
            section_path: Some(vec!["intro".into()]),
            meta: Default::default(),
            extra: Default::default(),
        },
    ]
}

fn ensure_parent_dir(db_path: &str) -> std::io::Result<()> {
    if let Some(parent) = std::path::Path::new(db_path).parent() { std::fs::create_dir_all(parent)?; }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    // Default DB path under target/demo to avoid cluttering workspace root
    let default_db = String::from("target/demo/chunks.db");
    let first = args.next();
    // If first arg starts with '-' or is None, treat as missing db path
    let (db_path, rest_start) = match first {
        Some(ref s) if s.starts_with('-') => (default_db.clone(), Some(s.clone())),
        Some(s) => (s, None),
        None => (default_db.clone(), None),
    };

    let mut ndjson: Option<String> = None;
    let mut use_sample = false;
    let mut search_query: Option<String> = None;
    let mut debug = false;

    // Recompose remaining args
    let mut tail: Vec<String> = Vec::new();
    if let Some(s) = rest_start { tail.push(s); }
    tail.extend(args);
    let rest: Vec<String> = tail;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--ndjson" => { if i + 1 < rest.len() { ndjson = Some(rest[i+1].clone()); i += 2; } else { print_usage(); return Ok(()); } }
            "--sample" => { use_sample = true; i += 1; }
            "--search" => { if i + 1 < rest.len() { search_query = Some(rest[i+1].clone()); i += 2; } else { print_usage(); return Ok(()); } }
            "--debug" => { debug = true; i += 1; }
            _ => { eprintln!("Unknown arg: {}", rest[i]); print_usage(); return Ok(()); }
        }
    }

    if ndjson.is_none() && !use_sample {
        eprintln!("Either --ndjson or --sample must be provided");
        print_usage();
        return Ok(());
    }

    ensure_parent_dir(&db_path)?;
    let mut repo = SqliteRepo::open(&db_path)?;
    let records = if let Some(p) = ndjson { load_ndjson(&p)? } else { make_sample() };

    // Upsert into DB and update text indexes (FTS5 maintainer is no-op; Tantivy can be added later)
    let fts = Fts5Index::new();
    let text_indexes: [&dyn chunking_store::TextIndexMaintainer; 1] = [&fts];
    let mut vector_indexes: [&mut dyn chunking_store::VectorIndexMaintainer; 0] = [];
    ingest_chunks_orchestrated(&mut repo, &records, &text_indexes, &mut vector_indexes, None)
        .map_err(|e| -> Box<dyn std::error::Error> { Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("{e}"))) })?;

    println!("Ingested {} chunk(s)", records.len());

    if debug {
        if let Ok((chunks_cnt, fts_cnt)) = repo.counts() {
            println!("Counts before rebuild check: chunks={}, fts={}", chunks_cnt, fts_cnt);
        }
        let _ = repo.maybe_rebuild_fts();
        if let Ok((chunks_cnt, fts_cnt)) = repo.counts() {
            println!("Counts after rebuild check: chunks={}, fts={}", chunks_cnt, fts_cnt);
        }
        if let Ok(n) = repo.fts_match_count("hello") {
            println!("FTS MATCH 'hello' count = {}", n);
        }
    }

    if let Some(q) = search_query {
        let hits = fts.search_simple(&repo, &q, 10);
        println!("Search '{}' -> {} hit(s)", q, hits.len());
        for h in hits {
            let id = (h.chunk.chunk_id).0;
            let text = &h.chunk.text;
            let preview = if text.len() > 60 { &text[..60] } else { text };
            println!("- {} score={:.4} text='{}'", id, h.score, preview);
        }
    }

    Ok(())
}
