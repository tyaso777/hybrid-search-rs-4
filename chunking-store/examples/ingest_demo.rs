use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

use chunk_model::{ChunkRecord, DocumentId, ChunkId, SCHEMA_MAJOR};
use chunking_store::orchestrator::ingest_chunks_orchestrated;
use chunking_store::sqlite_repo::SqliteRepo;

fn print_usage() {
    eprintln!(
        "Usage: ingest_demo [db_path] [--ndjson PATH | --sample]\n\
         Examples:\n\
           ingest_demo                --sample                (uses target/demo/chunks.db)\n\
           ingest_demo ./chunks.db    --ndjson ./chunks.ndjson\n"
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
            page_start: None,
            page_end: None,
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
            page_start: None,
            page_end: None,
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

    // Upsert into DB (no text indexes configured by default)
    let text_indexes: [&dyn chunking_store::TextIndexMaintainer; 0] = [];
    let mut vector_indexes: [&mut dyn chunking_store::VectorIndexMaintainer; 0] = [];
    ingest_chunks_orchestrated(&mut repo, &records, &text_indexes, &mut vector_indexes, None)
        .map_err(|e| -> Box<dyn std::error::Error> { Box::new(std::io::Error::new(std::io::ErrorKind::Other, format!("{e}"))) })?;

    println!("Ingested {} chunk(s)", records.len());

    Ok(())
}
