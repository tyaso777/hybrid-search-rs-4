use std::env;
use std::path::Path;

use chunking_store::fts5_index::Fts5Index;
use chunking_store::orchestrator::delete_by_filter_orchestrated;
use chunking_store::sqlite_repo::SqliteRepo;
use chunking_store::{FilterClause, FilterKind, FilterOp};

fn print_usage() {
    eprintln!(
        "Usage: delete_demo [db_path] [--doc-id DOC] [--prefix URI_PREFIX] [--start ISO] [--end ISO] [--batch N]\n\
         Examples:\n\
           delete_demo                   --doc-id doc-001   (uses target/demo/chunks.db)\n\
           delete_demo ./chunks.db       --prefix file:///data/ --start 2024-01-01T00:00:00Z --end 2025-01-01T00:00:00Z\n"
    );
}

fn main() {
    let mut args = env::args().skip(1);
    let default_db = String::from("target/demo/chunks.db");
    let first = args.next();
    let (db_path, rest_start) = match first {
        Some(ref s) if s.starts_with('-') => (default_db.clone(), Some(s.clone())),
        Some(s) => (s, None),
        None => (default_db.clone(), None),
    };

    let mut doc_id: Option<String> = None;
    let mut prefix: Option<String> = None;
    let mut start: Option<String> = None;
    let mut end: Option<String> = None;
    let mut batch_size: usize = 1000;

    let mut tail: Vec<String> = Vec::new();
    if let Some(s) = rest_start { tail.push(s); }
    tail.extend(args);
    let rest: Vec<String> = tail;
    let mut i = 0;
    while i < rest.len() {
        match rest[i].as_str() {
            "--doc-id" => { if i + 1 < rest.len() { doc_id = Some(rest[i+1].clone()); i += 2; } else { print_usage(); return; } }
            "--prefix" => { if i + 1 < rest.len() { prefix = Some(rest[i+1].clone()); i += 2; } else { print_usage(); return; } }
            "--start" => { if i + 1 < rest.len() { start = Some(rest[i+1].clone()); i += 2; } else { print_usage(); return; } }
            "--end" => { if i + 1 < rest.len() { end = Some(rest[i+1].clone()); i += 2; } else { print_usage(); return; } }
            "--batch" => { if i + 1 < rest.len() { batch_size = rest[i+1].parse().unwrap_or(1000); i += 2; } else { print_usage(); return; } }
            _ => { eprintln!("Unknown arg: {}", rest[i]); print_usage(); return; }
        }
    }

    let mut filters: Vec<FilterClause> = Vec::new();
    if let Some(d) = doc_id { filters.push(FilterClause { kind: FilterKind::Must, op: FilterOp::DocIdEq(d) }); }
    if let Some(p) = prefix { filters.push(FilterClause { kind: FilterKind::Must, op: FilterOp::SourceUriPrefix(p) }); }
    if start.is_some() || end.is_some() {
        filters.push(FilterClause { kind: FilterKind::Must, op: FilterOp::RangeIsoDate { key: "extracted_at".into(), start, end, start_incl: true, end_incl: false } });
    }

    if filters.is_empty() {
        eprintln!("No filters specified. Refusing to delete everything. Provide at least one filter.");
        print_usage();
        return;
    }

    if let Some(parent) = Path::new(&db_path).parent() { let _ = std::fs::create_dir_all(parent); }
    let mut repo = SqliteRepo::open(&db_path).expect("open sqlite repo");

    // FTS5 maintainer is no-op (DB triggers keep it in sync), but we pass it to the orchestrator for symmetry.
    let fts = Fts5Index::new();
    let text_indexes: [&dyn chunking_store::TextIndexMaintainer; 1] = [&fts];
    let mut vector_indexes: [&mut dyn chunking_store::VectorIndexMaintainer; 0] = [];

    let report = delete_by_filter_orchestrated(
        &mut repo,
        &filters,
        batch_size,
        &text_indexes,
        &mut vector_indexes,
    ).expect("orchestrated delete");

    println!(
        "Delete completed: total_ids={}, db_deleted={}, text_delete_attempts={}, vector_delete_attempts={}, batches={}",
        report.total_ids, report.db_deleted, report.text_delete_attempts, report.vector_delete_attempts, report.batches
    );
}
