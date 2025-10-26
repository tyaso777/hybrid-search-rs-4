use std::collections::HashMap;
use std::path::Path;

use chunk_model::{ChunkId, ChunkRecord, DocumentId};
use rusqlite::{params, Connection, TransactionBehavior};

use crate::{ChunkPrimaryStore, ChunkStoreRead, StoreError, FilterClause, FilterOp};

/// SQLite-backed primary store. FTS5 text search lives in `fts5_index`.
pub struct SqliteRepo {
    conn: Connection,
}

impl SqliteRepo {
    /// Open an in-memory repository and initialize schema.
    pub fn new() -> Self {
        let conn = Connection::open_in_memory().expect("open in-memory sqlite");
        let repo = Self { conn };
        repo.init().expect("initialize schema");
        repo
    }

    /// Open a file-backed repository at `path` and initialize schema if absent.
    pub fn open<P: AsRef<Path>>(path: P) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        let repo = Self { conn };
        repo.init()?;
        Ok(repo)
    }

    pub(crate) fn conn(&self) -> &Connection { &self.conn }

    fn init(&self) -> rusqlite::Result<()> {
        // Pragmas for durability and concurrency
        self.conn.pragma_update(None, "journal_mode", &"WAL")?;
        self.conn.pragma_update(None, "synchronous", &"FULL")?;
        self.conn.pragma_update(None, "foreign_keys", &"ON")?;

        // Core table for chunks
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS chunks (
                rowid INTEGER PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                chunk_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                source_uri TEXT NOT NULL,
                source_mime TEXT NOT NULL,
                extracted_at TEXT NOT NULL,
                text TEXT NOT NULL,
                section_path_json TEXT NOT NULL,
                meta_json TEXT NOT NULL,
                extra_json TEXT NOT NULL,
                vector BLOB
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

            -- FTS5 virtual table linked to chunks via content= and rowid
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='rowid',
                tokenize = 'unicode61'
            );

            -- Triggers to keep FTS index consistent
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF text ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
            END;
            "#,
        )?;
        Ok(())
    }

    /// Ensure FTS content table is populated; rebuild if empty while chunks has rows.
    pub fn maybe_rebuild_fts(&self) -> rusqlite::Result<()> {
        let chunks_cnt: i64 = self.conn.query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))?;
        if chunks_cnt == 0 { return Ok(()); }
        let fts_cnt: i64 = self.conn.query_row("SELECT count(*) FROM chunks_fts", [], |r| r.get(0)).unwrap_or(0);
        if fts_cnt == 0 {
            // Rebuild FTS index from content table
            let _ = self.conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')", []);
        }
        Ok(())
    }

    /// Return (chunks_count, chunks_fts_count) for debugging.
    pub fn counts(&self) -> rusqlite::Result<(i64, i64)> {
        let chunks_cnt: i64 = self.conn.query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))?;
        let fts_cnt: i64 = self.conn.query_row("SELECT count(*) FROM chunks_fts", [], |r| r.get(0)).unwrap_or(0);
        Ok((chunks_cnt, fts_cnt))
    }

    /// Return count of rows matching an FTS5 MATCH query (for debugging).
    pub fn fts_match_count(&self, query: &str) -> rusqlite::Result<i64> {
        self.conn.query_row(
            "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH ?1",
            [query],
            |r| r.get(0),
        )
    }

    /// List chunk IDs matching filters with pagination.
    pub fn list_chunk_ids_by_filter(
        &self,
        filters: &[crate::FilterClause],
        limit: usize,
        offset: usize,
    ) -> Result<Vec<ChunkId>, StoreError> {
        let mut where_sql = String::from("WHERE 1=1");
        let mut params: Vec<rusqlite::types::Value> = Vec::new();

        for f in filters {
            match &f.op {
                crate::FilterOp::DocIdEq(v) => {
                    where_sql.push_str(" AND doc_id = ?");
                    params.push(v.clone().into());
                }
                crate::FilterOp::DocIdIn(vs) => {
                    if !vs.is_empty() {
                        where_sql.push_str(" AND doc_id IN (");
                        for i in 0..vs.len() {
                            if i > 0 { where_sql.push(','); }
                            where_sql.push('?');
                            params.push(vs[i].clone().into());
                        }
                        where_sql.push(')');
                    }
                }
                crate::FilterOp::SourceUriPrefix(p) => {
                    where_sql.push_str(" AND source_uri LIKE ?");
                    params.push(format!("{}%", p).into());
                }
                // Range on extracted_at (ISO 8601 strings) using lexicographic compare
                crate::FilterOp::RangeIsoDate { key, start, end, start_incl, end_incl } => {
                    if key == "extracted_at" {
                        if let Some(s) = start {
                            if *start_incl {
                                where_sql.push_str(" AND extracted_at >= ?");
                            } else {
                                where_sql.push_str(" AND extracted_at > ?");
                            }
                            params.push(s.clone().into());
                        }
                        if let Some(e) = end {
                            if *end_incl {
                                where_sql.push_str(" AND extracted_at <= ?");
                            } else {
                                where_sql.push_str(" AND extracted_at < ?");
                            }
                            params.push(e.clone().into());
                        }
                    }
                }
                // Meta equality via JSON1
                crate::FilterOp::MetaEq { key, value } => {
                    where_sql.push_str(" AND json_extract(meta_json, ?) = ?");
                    let path = format!("$.\"{}\"", key.replace('"', "\""));
                    params.push(path.into());
                    params.push(value.clone().into());
                }
                // Meta IN via JSON1
                crate::FilterOp::MetaIn { key, values } => {
                    if !values.is_empty() {
                        where_sql.push_str(" AND json_extract(meta_json, ?) IN (");
                        let path = format!("$.\"{}\"", key.replace('"', "\""));
                        params.push(path.into());
                        for i in 0..values.len() {
                            if i > 0 { where_sql.push(','); }
                            where_sql.push('?');
                            params.push(values[i].clone().into());
                        }
                        where_sql.push(')');
                    }
                }
                // Numeric range on meta via JSON1 + CAST
                crate::FilterOp::RangeNumeric { key, min, max, min_incl, max_incl } => {
                    if let Some(lo) = min {
                        where_sql.push_str(" AND CAST(json_extract(meta_json, ?) AS REAL) ");
                        if *min_incl { where_sql.push_str(">= ?"); } else { where_sql.push_str("> ?"); }
                        let path = format!("$.\"{}\"", key.replace('"', "\""));
                        params.push(path.into());
                        params.push((*lo as f64).into());
                    }
                    if let Some(hi) = max {
                        where_sql.push_str(" AND CAST(json_extract(meta_json, ?) AS REAL) ");
                        if *max_incl { where_sql.push_str("<= ?"); } else { where_sql.push_str("< ?"); }
                        let path = format!("$.\"{}\"", key.replace('"', "\""));
                        params.push(path.into());
                        params.push((*hi as f64).into());
                    }
                }
            }
        }

        let sql = format!(
            "SELECT chunk_id FROM chunks {} ORDER BY rowid LIMIT ? OFFSET ?",
            where_sql
        );
        params.push((limit as i64).into());
        params.push((offset as i64).into());

        let mut stmt = self.conn
            .prepare(&sql)
            .map_err(|e| StoreError::Backend(e.to_string()))?;

        let rows = stmt
            .query_map(rusqlite::params_from_iter(params.into_iter()), |row| {
                let cid: String = row.get(0)?;
                Ok(ChunkId(cid))
            })
            .map_err(|e| StoreError::Backend(e.to_string()))?;

        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| StoreError::Backend(e.to_string()))?);
        }
        Ok(out)
    }
}

impl ChunkPrimaryStore for SqliteRepo {
    fn upsert_chunks(&mut self, chunks: Vec<ChunkRecord>) -> Result<(), StoreError> {
        if chunks.is_empty() {
            return Ok(());
        }

        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| StoreError::Backend(e.to_string()))?;

        let mut stmt = tx
            .prepare(
                r#"
            INSERT INTO chunks (
                schema_version, chunk_id, doc_id, source_uri, source_mime, extracted_at,
                text, section_path_json, meta_json, extra_json, vector
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, NULL)
            ON CONFLICT(chunk_id) DO UPDATE SET
                schema_version=excluded.schema_version,
                doc_id=excluded.doc_id,
                source_uri=excluded.source_uri,
                source_mime=excluded.source_mime,
                extracted_at=excluded.extracted_at,
                text=excluded.text,
                section_path_json=excluded.section_path_json,
                meta_json=excluded.meta_json,
                extra_json=excluded.extra_json
            ;
            "#,
            )
            .map_err(|e| StoreError::Backend(e.to_string()))?;

        for rec in chunks {
            if rec.validate_soft().is_err() {
                continue;
            }
            let section_json = match serde_json::to_string(&rec.section_path) {
                Ok(s) => s,
                Err(e) => return Err(StoreError::Backend(e.to_string())),
            };
            let meta_json = match serde_json::to_string(&rec.meta) {
                Ok(s) => s,
                Err(e) => return Err(StoreError::Backend(e.to_string())),
            };
            let extra_json = match serde_json::to_string(&rec.extra) {
                Ok(s) => s,
                Err(e) => return Err(StoreError::Backend(e.to_string())),
            };

            stmt
                .execute(params![
                    rec.schema_version as i64,
                    rec.chunk_id.0,
                    rec.doc_id.0,
                    rec.source_uri,
                    rec.source_mime,
                    rec.extracted_at,
                    rec.text,
                    section_json,
                    meta_json,
                    extra_json,
                ])
                .map_err(|e| StoreError::Backend(e.to_string()))?;
        }
        drop(stmt);
        tx.commit()
            .map_err(|e| StoreError::Backend(e.to_string()))
    }

    fn delete_by_ids(&mut self, ids: &[chunk_model::ChunkId]) -> Result<usize, StoreError> {
        if ids.is_empty() { return Ok(0); }
        let tx = self.conn.transaction().map_err(|e| StoreError::Backend(e.to_string()))?;
        let mut placeholders = String::from("(");
        for i in 0..ids.len() { if i > 0 { placeholders.push(','); } placeholders.push('?'); }
        placeholders.push(')');
        let sql = format!("DELETE FROM chunks WHERE chunk_id IN {}", placeholders);
        let params: Vec<&str> = ids.iter().map(|c| c.0.as_str()).collect();
        let n = tx.execute(&sql, rusqlite::params_from_iter(params.iter()))
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        tx.commit().map_err(|e| StoreError::Backend(e.to_string()))?;
        Ok(n)
    }

    fn delete_by_filter(&mut self, filters: &[FilterClause]) -> Result<usize, StoreError> {
        if filters.is_empty() { return Ok(0); }
        // Support a subset: DocIdEq/In, SourceUriPrefix, and some meta/date-range combined with AND
        let mut where_sql = String::from("WHERE 1=1");
        let mut params: Vec<rusqlite::types::Value> = Vec::new();
        for f in filters {
            match &f.op {
                FilterOp::DocIdEq(v) => { where_sql.push_str(" AND doc_id = ?"); params.push(v.clone().into()); }
                FilterOp::DocIdIn(vs) => {
                    if !vs.is_empty() {
                        where_sql.push_str(" AND doc_id IN (");
                        for i in 0..vs.len() { if i>0 { where_sql.push(','); } where_sql.push('?'); params.push(vs[i].clone().into()); }
                        where_sql.push(')');
                    }
                }
                FilterOp::SourceUriPrefix(p) => { where_sql.push_str(" AND source_uri LIKE ?"); params.push(format!("{}%", p).into()); }
                FilterOp::MetaEq { key, value } => {
                    where_sql.push_str(" AND json_extract(meta_json, ?) = ?");
                    let path = format!("$.\"{}\"", key.replace('"', "\""));
                    params.push(path.into());
                    params.push(value.clone().into());
                }
                FilterOp::MetaIn { key, values } => {
                    if !values.is_empty() {
                        where_sql.push_str(" AND json_extract(meta_json, ?) IN (");
                        let path = format!("$.\"{}\"", key.replace('"', "\""));
                        params.push(path.into());
                        for i in 0..values.len() { if i>0 { where_sql.push(','); } where_sql.push('?'); params.push(values[i].clone().into()); }
                        where_sql.push(')');
                    }
                }
                FilterOp::RangeIsoDate { key, start, end, start_incl, end_incl } => {
                    if key == "extracted_at" {
                        if let Some(s) = start { where_sql.push_str(if *start_incl {" AND extracted_at >= ?"} else {" AND extracted_at > ?"}); params.push(s.clone().into()); }
                        if let Some(e) = end { where_sql.push_str(if *end_incl {" AND extracted_at <= ?"} else {" AND extracted_at < ?"}); params.push(e.clone().into()); }
                    }
                }
                FilterOp::RangeNumeric { key, min, max, min_incl, max_incl } => {
                    if let Some(lo) = min { where_sql.push_str(" AND CAST(json_extract(meta_json, ?) AS REAL) "); where_sql.push_str(if *min_incl { ">= ?" } else { "> ?" }); let path = format!("$.\"{}\"", key.replace('"', "\"")); params.push(path.into()); params.push((*lo as f64).into()); }
                    if let Some(hi) = max { where_sql.push_str(" AND CAST(json_extract(meta_json, ?) AS REAL) "); where_sql.push_str(if *max_incl { "<= ?" } else { "< ?" }); let path = format!("$.\"{}\"", key.replace('"', "\"")); params.push(path.into()); params.push((*hi as f64).into()); }
                }
            }
        }
        let sql = format!("DELETE FROM chunks {}", where_sql);
        let n = self.conn.execute(&sql, rusqlite::params_from_iter(params.into_iter()))
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        Ok(n)
    }
}

impl ChunkStoreRead for SqliteRepo {
    fn get_chunks_by_ids(&self, ids: &[ChunkId]) -> Result<Vec<ChunkRecord>, StoreError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        // Build `IN (?, ?, ...)` placeholder list
        let mut placeholders = String::new();
        placeholders.push('(');
        for i in 0..ids.len() {
            if i > 0 { placeholders.push(','); }
            placeholders.push('?');
        }
        placeholders.push(')');

        let sql = format!(
            "SELECT schema_version, chunk_id, doc_id, source_uri, source_mime, extracted_at, text, section_path_json, meta_json, extra_json FROM chunks WHERE chunk_id IN {}",
            placeholders
        );

        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| StoreError::Backend(e.to_string()))?;

        // Bind parameters
        let params_vec: Vec<&str> = ids.iter().map(|c| c.0.as_str()).collect();
        let mut map: HashMap<String, ChunkRecord> = HashMap::with_capacity(ids.len());

        let rows = stmt
            .query_map(rusqlite::params_from_iter(params_vec.iter()), |row| {
                let schema_version: i64 = row.get(0)?;
                let chunk_id: String = row.get(1)?;
                let doc_id: String = row.get(2)?;
                let source_uri: String = row.get(3)?;
                let source_mime: String = row.get(4)?;
                let extracted_at: String = row.get(5)?;
                let text: String = row.get(6)?;
                let section_path_json: String = row.get(7)?;
                let meta_json: String = row.get(8)?;
                let extra_json: String = row.get(9)?;

                let section_path: Vec<String> = serde_json::from_str(&section_path_json).unwrap_or_default();
                let meta: std::collections::BTreeMap<String, String> = serde_json::from_str(&meta_json).unwrap_or_default();
                let extra: std::collections::BTreeMap<String, serde_json::Value> = serde_json::from_str(&extra_json).unwrap_or_default();

                Ok(ChunkRecord {
                    schema_version: schema_version as u16,
                    doc_id: DocumentId(doc_id.clone()),
                    chunk_id: ChunkId(chunk_id.clone()),
                    source_uri,
                    source_mime,
                    extracted_at,
                    text,
                    section_path,
                    meta,
                    extra,
                })
            })
            .map_err(|e| StoreError::Backend(e.to_string()))?;

        for r in rows {
            let rec = r.map_err(|e| StoreError::Backend(e.to_string()))?;
            map.insert(rec.chunk_id.0.clone(), rec);
        }

        // Preserve requested order
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(rec) = map.remove(&id.0) { out.push(rec); }
        }
        Ok(out)
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
}

