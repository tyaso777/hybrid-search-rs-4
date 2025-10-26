use chunk_model::{ChunkId, ChunkRecord};

use crate::sqlite_repo::SqliteRepo;
use crate::{SearchHit, TextMatch, ChunkStoreRead, TextSearcher, FilterClause, FilterKind, FilterOp, SearchOptions, IndexCaps};

/// FTS5-backed text search over the SQLite primary store.
/// Index maintenance is handled by SQLite triggers in the store.
#[derive(Default)]
pub struct Fts5Index;

impl Fts5Index {
    pub fn new() -> Self { Self }

    /// Convenience search (no filters) with defaults.
    pub fn search_simple(&self, repo: &SqliteRepo, query: &str, limit: usize) -> Vec<SearchHit> {
        let opts = SearchOptions { top_k: limit, fetch_factor: 10 };
        self.search(repo, query, &[], &opts)
    }

    /// Full search with filters and options; resolves IDs to full records and applies post-filters.
    pub fn search(&self, repo: &SqliteRepo, query: &str, filters: &[FilterClause], opts: &SearchOptions) -> Vec<SearchHit> {
        if query.trim().is_empty() || opts.top_k == 0 { return Vec::new(); }
        // Fetch IDs first (with pre-filters applied by the index), then materialize and post-filter.
        let matches = TextSearcher::search_ids(self, repo, query, filters, opts);
        if matches.is_empty() { return Vec::new(); }
        let ids: Vec<ChunkId> = matches.iter().map(|m| m.chunk_id.clone()).collect();
        let recs = match repo.get_chunks_by_ids(&ids) { Ok(r) => r, Err(_) => return Vec::new() };

        // Build quick lookup of score by id
        let mut score_map: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        for m in &matches { score_map.insert(m.chunk_id.0.clone(), m.score); }

        // Split filters into pre/post and apply post only here
        let (_pre, post) = plan_filters(self, filters);
        let mut hits = Vec::with_capacity(recs.len());
        for rec in recs {
            if !matches_filters(&rec, &post) { continue; }
            if let Some(score) = score_map.get(&rec.chunk_id.0) {
                hits.push(SearchHit { chunk: rec, score: *score });
            }
        }
        // Preserve ordering of matches
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(opts.top_k);
        hits
    }
}

impl TextSearcher for Fts5Index {
    fn name(&self) -> &'static str { "fts5" }

    fn caps(&self) -> IndexCaps {
        IndexCaps {
            can_prefilter_doc_id_eq: true,
            can_prefilter_doc_id_in: true,
            can_prefilter_source_prefix: true,
            can_prefilter_meta: false,
        }
    }

    fn search_ids(
        &self,
        store: &dyn ChunkStoreRead,
        query: &str,
        filters: &[FilterClause],
        opts: &SearchOptions,
    ) -> Vec<TextMatch> {
        let Some(sqlite) = any_as_sqlite(store) else { return Vec::new(); };
        if query.trim().is_empty() || opts.top_k == 0 { return Vec::new(); }

        let (pre, _post) = plan_filters(self, filters);

        // Build SQL dynamically with pre-filters
        let mut sql = String::from(
            "SELECT c.chunk_id, bm25(f) as rank \n\
             FROM chunks_fts f \n\
             JOIN chunks c ON c.rowid = f.rowid \n\
             WHERE f MATCH ?1",
        );
        let mut params: Vec<rusqlite::types::Value> = vec![rusqlite::types::Value::from(query.to_string())];

        // doc_id = ? or IN (...)
        for fc in &pre {
            match &fc.op {
                FilterOp::DocIdEq(v) => {
                    sql.push_str(" AND c.doc_id = ?");
                    params.push(v.clone().into());
                }
                FilterOp::DocIdIn(vs) => {
                    if !vs.is_empty() {
                        sql.push_str(" AND c.doc_id IN (");
                        for i in 0..vs.len() {
                            if i > 0 { sql.push(','); }
                            sql.push('?');
                            params.push(vs[i].clone().into());
                        }
                        sql.push(')');
                    }
                }
                FilterOp::SourceUriPrefix(prefix) => {
                    sql.push_str(" AND c.source_uri LIKE ?");
                    params.push(format!("{}%", prefix).into());
                }
                _ => {}
            }
        }

        sql.push_str(" ORDER BY rank LIMIT ?");
        let fetch_n = (opts.top_k.saturating_mul(opts.fetch_factor)).max(opts.top_k);
        params.push((fetch_n as i64).into());

        let conn = sqlite.conn();
        let mut stmt = match conn.prepare(&sql) { Ok(s) => s, Err(_) => return Vec::new() };
        let rows = match stmt.query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
            let chunk_id: String = row.get(0)?;
            let rank: f64 = row.get(1)?; // smaller is better
            // normalize so larger is better
            let score = 1.0f32 / (1.0f32 + (rank as f32));
            Ok(TextMatch { chunk_id: ChunkId(chunk_id), score })
        }) { Ok(r) => r, Err(_) => return Vec::new() };

        let mut out = Vec::new();
        for r in rows { if let Ok(m) = r { out.push(m); } }
        out
    }
}

fn plan_filters<'a>(idx: &impl TextSearcher, filters: &'a [FilterClause]) -> (Vec<FilterClause>, Vec<FilterClause>) {
    let caps = idx.caps();
    let mut pre = Vec::new();
    let mut post = Vec::new();
    for f in filters {
        let supported = match &f.op {
            FilterOp::DocIdEq(_) => caps.can_prefilter_doc_id_eq,
            FilterOp::DocIdIn(_) => caps.can_prefilter_doc_id_in,
            FilterOp::SourceUriPrefix(_) => caps.can_prefilter_source_prefix,
            FilterOp::MetaEq { .. } | FilterOp::MetaIn { .. } => caps.can_prefilter_meta,
        };
        if supported && f.kind != FilterKind::PostOnly {
            pre.push(f.clone());
        } else {
            post.push(f.clone());
        }
    }
    (pre, post)
}

fn matches_filters(rec: &ChunkRecord, post: &[FilterClause]) -> bool {
    'outer: for f in post {
        match &f.op {
            FilterOp::DocIdEq(v) => { if &rec.doc_id.0 != v { continue 'outer; } }
            FilterOp::DocIdIn(vs) => { if !vs.iter().any(|v| v == &rec.doc_id.0) { continue 'outer; } }
            FilterOp::SourceUriPrefix(prefix) => { if !rec.source_uri.starts_with(prefix) { continue 'outer; } }
            FilterOp::MetaEq { key, value } => {
                match rec.meta.get(key) { Some(v) if v == value => {}, _ => continue 'outer }
            }
            FilterOp::MetaIn { key, values } => {
                match rec.meta.get(key) { Some(v) if values.iter().any(|x| x == v) => {}, _ => continue 'outer }
            }
        }
    }
    true
}

fn any_as_sqlite(store: &dyn ChunkStoreRead) -> Option<&SqliteRepo> {
    store.as_any().downcast_ref::<SqliteRepo>()
}

