use std::collections::HashMap;
use std::path::Path;

use chunk_model::{ChunkId, ChunkRecord, DocumentId, FileRecord};
use serde_json::Value as JsonValue;
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
                page_start INTEGER,
                page_end INTEGER,
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

            -- File-level table to persist FileRecord (one row per doc_id)
            CREATE TABLE IF NOT EXISTS files (
                doc_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                doc_revision INTEGER,
                source_uri TEXT NOT NULL,
                source_mime TEXT NOT NULL,
                file_size_bytes INTEGER,
                content_sha256 TEXT,
                page_count INTEGER,
                extracted_at TEXT NOT NULL,
                created_at_meta TEXT,
                updated_at_meta TEXT,
                title_guess TEXT,
                author_guess TEXT,
                dominant_lang TEXT,
                tags_json TEXT NOT NULL,
                ingest_tool TEXT,
                ingest_tool_version TEXT,
                reader_backend TEXT,
                ocr_used INTEGER,
                ocr_langs_json TEXT NOT NULL,
                chunk_count INTEGER,
                total_tokens INTEGER,
                meta_json TEXT NOT NULL,
                extra_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_files_source_uri ON files(source_uri);
            "#,
        )?;
        // Best-effort migration for older tables missing page_start/page_end
        let _ = self.conn.execute("ALTER TABLE chunks ADD COLUMN page_start INTEGER", []);
        let _ = self.conn.execute("ALTER TABLE chunks ADD COLUMN page_end INTEGER", []);
        Ok(())
    }

    /// Upsert one FileRecord into the files table keyed by doc_id.
    pub fn upsert_file(&self, file: &FileRecord) -> rusqlite::Result<()> {
        let tags_json = serde_json::to_string(&file.tags).unwrap_or_else(|_| "[]".to_string());
        let ocr_langs_json = serde_json::to_string(&file.ocr_langs).unwrap_or_else(|_| "[]".to_string());
        let meta_json = serde_json::to_string(&file.meta).unwrap_or_else(|_| "{}".to_string());
        let extra_json = serde_json::to_string(&file.extra).unwrap_or_else(|_| "{}".to_string());
        self.conn
            .execute(
                r#"
                INSERT INTO files (
                    doc_id, schema_version, doc_revision, source_uri, source_mime,
                    file_size_bytes, content_sha256, page_count, extracted_at, created_at_meta,
                    updated_at_meta, title_guess, author_guess, dominant_lang, tags_json,
                    ingest_tool, ingest_tool_version, reader_backend, ocr_used, ocr_langs_json,
                    chunk_count, total_tokens, meta_json, extra_json
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24)
                ON CONFLICT(doc_id) DO UPDATE SET
                    schema_version=excluded.schema_version,
                    doc_revision=excluded.doc_revision,
                    source_uri=excluded.source_uri,
                    source_mime=excluded.source_mime,
                    file_size_bytes=excluded.file_size_bytes,
                    content_sha256=excluded.content_sha256,
                    page_count=excluded.page_count,
                    extracted_at=excluded.extracted_at,
                    created_at_meta=excluded.created_at_meta,
                    updated_at_meta=excluded.updated_at_meta,
                    title_guess=excluded.title_guess,
                    author_guess=excluded.author_guess,
                    dominant_lang=excluded.dominant_lang,
                    tags_json=excluded.tags_json,
                    ingest_tool=excluded.ingest_tool,
                    ingest_tool_version=excluded.ingest_tool_version,
                    reader_backend=excluded.reader_backend,
                    ocr_used=excluded.ocr_used,
                    ocr_langs_json=excluded.ocr_langs_json,
                    chunk_count=excluded.chunk_count,
                    total_tokens=excluded.total_tokens,
                    meta_json=excluded.meta_json,
                    extra_json=excluded.extra_json
                ;
                "#,
                params![
                    file.doc_id.0,
                    file.schema_version as i64,
                    file.doc_revision.map(|v| v as i64),
                    file.source_uri,
                    file.source_mime,
                    file.file_size_bytes.map(|v| v as i64),
                    file.content_sha256,
                    file.page_count.map(|v| v as i64),
                    file.extracted_at,
                    file.created_at_meta,
                    file.updated_at_meta,
                    file.title_guess,
                    file.author_guess,
                    file.dominant_lang,
                    tags_json,
                    file.ingest_tool,
                    file.ingest_tool_version,
                    file.reader_backend,
                    file.ocr_used.map(|b| if b { 1i64 } else { 0i64 }),
                    ocr_langs_json,
                    file.chunk_count.map(|v| v as i64),
                    file.total_tokens.map(|v| v as i64),
                    meta_json,
                    extra_json,
                ],
            )?
            ;
        Ok(())
    }

    /// List FileRecords with pagination.
    pub fn list_files(&self, limit: usize, offset: usize) -> rusqlite::Result<Vec<FileRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT doc_id, schema_version, doc_revision, source_uri, source_mime, file_size_bytes, content_sha256, page_count, extracted_at, created_at_meta, updated_at_meta, title_guess, author_guess, dominant_lang, tags_json, ingest_tool, ingest_tool_version, reader_backend, ocr_used, ocr_langs_json, chunk_count, total_tokens, meta_json, extra_json FROM files ORDER BY extracted_at DESC LIMIT ?1 OFFSET ?2"
        )?;
        let rows = stmt.query_map(params![limit as i64, offset as i64], |row| {
            let doc_id: String = row.get(0)?;
            let schema_version: i64 = row.get(1)?;
            let doc_revision: Option<i64> = row.get(2).ok();
            let source_uri: String = row.get(3)?;
            let source_mime: String = row.get(4)?;
            let file_size_bytes: Option<i64> = row.get(5).ok();
            let content_sha256: Option<String> = row.get(6).ok();
            let page_count: Option<i64> = row.get(7).ok();
            let extracted_at: String = row.get(8)?;
            let created_at_meta: Option<String> = row.get(9).ok();
            let updated_at_meta: Option<String> = row.get(10).ok();
            let title_guess: Option<String> = row.get(11).ok();
            let author_guess: Option<String> = row.get(12).ok();
            let dominant_lang: Option<String> = row.get(13).ok();
            let tags_json: String = row.get(14)?;
            let ingest_tool: Option<String> = row.get(15).ok();
            let ingest_tool_version: Option<String> = row.get(16).ok();
            let reader_backend: Option<String> = row.get(17).ok();
            let ocr_used_opt: Option<i64> = row.get(18).ok();
            let ocr_langs_json: String = row.get(19)?;
            let chunk_count: Option<i64> = row.get(20).ok();
            let total_tokens: Option<i64> = row.get(21).ok();
            let meta_json: String = row.get(22)?;
            let extra_json: String = row.get(23)?;

            let tags: Vec<String> = serde_json::from_str(&tags_json).unwrap_or_default();
            let ocr_langs: Vec<String> = serde_json::from_str(&ocr_langs_json).unwrap_or_default();
            let meta: std::collections::BTreeMap<String, String> = serde_json::from_str(&meta_json).unwrap_or_default();
            let extra: std::collections::BTreeMap<String, JsonValue> = serde_json::from_str(&extra_json).unwrap_or_default();

            Ok(FileRecord {
                schema_version: schema_version as u16,
                doc_id: DocumentId(doc_id),
                doc_revision: doc_revision.and_then(|v| u32::try_from(v).ok()),
                source_uri,
                source_mime,
                file_size_bytes: file_size_bytes.and_then(|v| u64::try_from(v).ok()),
                content_sha256,
                page_count: page_count.and_then(|v| u32::try_from(v).ok()),
                extracted_at,
                created_at_meta,
                updated_at_meta,
                title_guess,
                author_guess,
                dominant_lang,
                tags,
                ingest_tool,
                ingest_tool_version,
                reader_backend,
                ocr_used: ocr_used_opt.map(|v| v != 0),
                ocr_langs,
                chunk_count: chunk_count.and_then(|v| u32::try_from(v).ok()),
                total_tokens: total_tokens.and_then(|v| u32::try_from(v).ok()),
                meta,
                extra,
            })
        })?;
        let mut out = Vec::new();
        for r in rows { out.push(r?); }
        Ok(out)
    }

    /// Delete files rows by doc_id list. Returns affected rows.
    pub fn delete_files_by_doc_ids(&self, doc_ids: &[String]) -> rusqlite::Result<usize> {
        if doc_ids.is_empty() { return Ok(0); }
        let mut placeholders = String::from("(");
        for i in 0..doc_ids.len() { if i > 0 { placeholders.push(','); } placeholders.push('?'); }
        placeholders.push(')');
        let sql = format!("DELETE FROM files WHERE doc_id IN {}", placeholders);
        let params: Vec<&str> = doc_ids.iter().map(|s| s.as_str()).collect();
        let n = self.conn.execute(&sql, rusqlite::params_from_iter(params.iter()))?;
        Ok(n)
    }

    /// Remove files that have no remaining chunks (best-effort cleanup).
    pub fn cleanup_orphan_files(&self) -> rusqlite::Result<usize> {
        let n = self.conn.execute(
            "DELETE FROM files WHERE doc_id NOT IN (SELECT DISTINCT doc_id FROM chunks)",
            [],
        )?;
        Ok(n)
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
                // Numeric range on columns (page_start/page_end) or meta via JSON1 + CAST
                crate::FilterOp::RangeNumeric { key, min, max, min_incl, max_incl } => {
                    let mut push_bound = |col: &str, is_min: bool, incl: bool, val: f64| {
                        where_sql.push_str(" AND ");
                        where_sql.push_str(col);
                        where_sql.push_str(" ");
                        if is_min { if incl { where_sql.push_str(">= ?"); } else { where_sql.push_str("> ?"); } }
                        else { if incl { where_sql.push_str("<= ?"); } else { where_sql.push_str("< ?"); } }
                        params.push(val.into());
                    };
                    match key.as_str() {
                        "page_start" => {
                            if let Some(lo) = min { push_bound("page_start", true, *min_incl, *lo as f64); }
                            if let Some(hi) = max { push_bound("page_start", false, *max_incl, *hi as f64); }
                        }
                        "page_end" => {
                            if let Some(lo) = min { push_bound("page_end", true, *min_incl, *lo as f64); }
                            if let Some(hi) = max { push_bound("page_end", false, *max_incl, *hi as f64); }
                        }
                        _ => {
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
                page_start, page_end,
                text, section_path_json, meta_json, extra_json, vector
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, NULL)
            ON CONFLICT(chunk_id) DO UPDATE SET
                schema_version=excluded.schema_version,
                doc_id=excluded.doc_id,
                source_uri=excluded.source_uri,
                source_mime=excluded.source_mime,
                extracted_at=excluded.extracted_at,
                page_start=excluded.page_start,
                page_end=excluded.page_end,
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
                    rec.page_start.map(|v| v as i64),
                    rec.page_end.map(|v| v as i64),
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
                    let push_bound = |sql: &mut String, col: &str, is_min: bool, incl: bool| {
                        sql.push_str(" AND ");
                        sql.push_str(col);
                        if is_min { if incl { sql.push_str(" >= ?"); } else { sql.push_str(" > ?"); } }
                        else { if incl { sql.push_str(" <= ?"); } else { sql.push_str(" < ?"); } }
                    };
                    match key.as_str() {
                        "page_start" => {
                            if let Some(lo) = min { push_bound(&mut where_sql, "page_start", true, *min_incl); params.push((*lo as f64).into()); }
                            if let Some(hi) = max { push_bound(&mut where_sql, "page_start", false, *max_incl); params.push((*hi as f64).into()); }
                        }
                        "page_end" => {
                            if let Some(lo) = min { push_bound(&mut where_sql, "page_end", true, *min_incl); params.push((*lo as f64).into()); }
                            if let Some(hi) = max { push_bound(&mut where_sql, "page_end", false, *max_incl); params.push((*hi as f64).into()); }
                        }
                        _ => {
                            if let Some(lo) = min {
                                where_sql.push_str(" AND CAST(json_extract(meta_json, ?) AS REAL) ");
                                where_sql.push_str(if *min_incl { ">= ?" } else { "> ?" });
                                let path = format!("$.\"{}\"", key.replace('"', "\""));
                                params.push(path.into());
                                params.push((*lo as f64).into());
                            }
                            if let Some(hi) = max {
                                where_sql.push_str(" AND CAST(json_extract(meta_json, ?) AS REAL) ");
                                where_sql.push_str(if *max_incl { "<= ?" } else { "< ?" });
                                let path = format!("$.\"{}\"", key.replace('"', "\""));
                                params.push(path.into());
                                params.push((*hi as f64).into());
                            }
                        }
                    }
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
            "SELECT schema_version, chunk_id, doc_id, source_uri, source_mime, extracted_at, page_start, page_end, text, section_path_json, meta_json, extra_json FROM chunks WHERE chunk_id IN {}",
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
                let page_start_opt: Option<i64> = row.get(6).ok();
                let page_end_opt: Option<i64> = row.get(7).ok();
                let text: String = row.get(8)?;
                let section_path_json: String = row.get(9)?;
                let meta_json: String = row.get(10)?;
                let extra_json: String = row.get(11)?;

                let section_path: Option<Vec<String>> = serde_json::from_str(&section_path_json).ok();
                let meta: std::collections::BTreeMap<String, String> = serde_json::from_str(&meta_json).unwrap_or_default();
                let extra: std::collections::BTreeMap<String, serde_json::Value> = serde_json::from_str(&extra_json).unwrap_or_default();

                Ok(ChunkRecord {
                    schema_version: schema_version as u16,
                    doc_id: DocumentId(doc_id.clone()),
                    chunk_id: ChunkId(chunk_id.clone()),
                    source_uri,
                    source_mime,
                    extracted_at,
                    page_start: page_start_opt.and_then(|v| u32::try_from(v).ok()),
                    page_end: page_end_opt.and_then(|v| u32::try_from(v).ok()),
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

impl SqliteRepo {
    /// Fetch a single chunk by its chunk_id.
    pub fn get_chunk_by_id(&self, id: &ChunkId) -> Result<Option<ChunkRecord>, StoreError> {
        let mut stmt = self
            .conn
            .prepare("SELECT schema_version, chunk_id, doc_id, source_uri, source_mime, extracted_at, page_start, page_end, text, section_path_json, meta_json, extra_json FROM chunks WHERE chunk_id = ?1")
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        let mut rows = stmt
            .query([id.0.as_str()])
            .map_err(|e| StoreError::Backend(e.to_string()))?;
        if let Some(row) = rows.next().map_err(|e| StoreError::Backend(e.to_string()))? {
            let schema_version: i64 = row.get(0).map_err(|e| StoreError::Backend(e.to_string()))?;
            let chunk_id: String = row.get(1).map_err(|e| StoreError::Backend(e.to_string()))?;
            let doc_id: String = row.get(2).map_err(|e| StoreError::Backend(e.to_string()))?;
            let source_uri: String = row.get(3).map_err(|e| StoreError::Backend(e.to_string()))?;
            let source_mime: String = row.get(4).map_err(|e| StoreError::Backend(e.to_string()))?;
            let extracted_at: String = row.get(5).map_err(|e| StoreError::Backend(e.to_string()))?;
            let page_start_opt: Option<i64> = row.get(6).ok();
            let page_end_opt: Option<i64> = row.get(7).ok();
            let text: String = row.get(8).map_err(|e| StoreError::Backend(e.to_string()))?;
            let section_path_json: String = row.get(9).map_err(|e| StoreError::Backend(e.to_string()))?;
            let meta_json: String = row.get(10).map_err(|e| StoreError::Backend(e.to_string()))?;
            let extra_json: String = row.get(11).map_err(|e| StoreError::Backend(e.to_string()))?;

            let section_path: Option<Vec<String>> = serde_json::from_str(&section_path_json).ok();
            let meta: std::collections::BTreeMap<String, String> = serde_json::from_str(&meta_json).unwrap_or_default();
            let extra: std::collections::BTreeMap<String, JsonValue> = serde_json::from_str(&extra_json).unwrap_or_default();

            Ok(Some(ChunkRecord {
                schema_version: schema_version as u16,
                doc_id: DocumentId(doc_id),
                chunk_id: ChunkId(chunk_id),
                source_uri,
                source_mime,
                extracted_at,
                page_start: page_start_opt.and_then(|v| u32::try_from(v).ok()),
                page_end: page_end_opt.and_then(|v| u32::try_from(v).ok()),
                text,
                section_path,
                meta,
                extra,
            }))
        } else {
            Ok(None)
        }
    }

    /// Return previous and next chunks within the same document, ordered by rowid.
    pub fn get_neighbor_chunks(&self, id: &ChunkId) -> Result<(Option<ChunkRecord>, Option<ChunkRecord>), StoreError> {
        // Find doc_id and rowid for the current chunk
        let (doc_id, rowid): (String, i64) = self
            .conn
            .query_row(
                "SELECT doc_id, rowid FROM chunks WHERE chunk_id = ?1",
                [id.0.as_str()],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(|e| StoreError::Backend(e.to_string()))?;

        // Previous
        let prev: Option<ChunkRecord> = {
            let mut stmt = self
                .conn
                .prepare("SELECT schema_version, chunk_id, doc_id, source_uri, source_mime, extracted_at, page_start, page_end, text, section_path_json, meta_json, extra_json FROM chunks WHERE doc_id = ?1 AND rowid < ?2 ORDER BY rowid DESC LIMIT 1")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let mut rows = stmt
                .query([doc_id.as_str(), &rowid.to_string()])
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            if let Some(row) = rows.next().map_err(|e| StoreError::Backend(e.to_string()))? {
                let schema_version: i64 = row.get(0).map_err(|e| StoreError::Backend(e.to_string()))?;
                let chunk_id: String = row.get(1).map_err(|e| StoreError::Backend(e.to_string()))?;
                let doc_id2: String = row.get(2).map_err(|e| StoreError::Backend(e.to_string()))?;
                let source_uri: String = row.get(3).map_err(|e| StoreError::Backend(e.to_string()))?;
                let source_mime: String = row.get(4).map_err(|e| StoreError::Backend(e.to_string()))?;
                let extracted_at: String = row.get(5).map_err(|e| StoreError::Backend(e.to_string()))?;
                let page_start_opt: Option<i64> = row.get(6).ok();
                let page_end_opt: Option<i64> = row.get(7).ok();
                let text: String = row.get(8).map_err(|e| StoreError::Backend(e.to_string()))?;
                let section_path_json: String = row.get(9).map_err(|e| StoreError::Backend(e.to_string()))?;
                let meta_json: String = row.get(10).map_err(|e| StoreError::Backend(e.to_string()))?;
                let extra_json: String = row.get(11).map_err(|e| StoreError::Backend(e.to_string()))?;
                let section_path: Option<Vec<String>> = serde_json::from_str(&section_path_json).ok();
                let meta: std::collections::BTreeMap<String, String> = serde_json::from_str(&meta_json).unwrap_or_default();
                let extra: std::collections::BTreeMap<String, JsonValue> = serde_json::from_str(&extra_json).unwrap_or_default();
                Some(ChunkRecord {
                    schema_version: schema_version as u16,
                    doc_id: DocumentId(doc_id2),
                    chunk_id: ChunkId(chunk_id),
                    source_uri,
                    source_mime,
                    extracted_at,
                    page_start: page_start_opt.and_then(|v| u32::try_from(v).ok()),
                    page_end: page_end_opt.and_then(|v| u32::try_from(v).ok()),
                    text,
                    section_path,
                    meta,
                    extra,
                })
            } else { None }
        };

        // Next
        let next: Option<ChunkRecord> = {
            let mut stmt = self
                .conn
                .prepare("SELECT schema_version, chunk_id, doc_id, source_uri, source_mime, extracted_at, page_start, page_end, text, section_path_json, meta_json, extra_json FROM chunks WHERE doc_id = ?1 AND rowid > ?2 ORDER BY rowid ASC LIMIT 1")
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            let mut rows = stmt
                .query([doc_id.as_str(), &rowid.to_string()])
                .map_err(|e| StoreError::Backend(e.to_string()))?;
            if let Some(row) = rows.next().map_err(|e| StoreError::Backend(e.to_string()))? {
                let schema_version: i64 = row.get(0).map_err(|e| StoreError::Backend(e.to_string()))?;
                let chunk_id: String = row.get(1).map_err(|e| StoreError::Backend(e.to_string()))?;
                let doc_id2: String = row.get(2).map_err(|e| StoreError::Backend(e.to_string()))?;
                let source_uri: String = row.get(3).map_err(|e| StoreError::Backend(e.to_string()))?;
                let source_mime: String = row.get(4).map_err(|e| StoreError::Backend(e.to_string()))?;
                let extracted_at: String = row.get(5).map_err(|e| StoreError::Backend(e.to_string()))?;
                let page_start_opt: Option<i64> = row.get(6).ok();
                let page_end_opt: Option<i64> = row.get(7).ok();
                let text: String = row.get(8).map_err(|e| StoreError::Backend(e.to_string()))?;
                let section_path_json: String = row.get(9).map_err(|e| StoreError::Backend(e.to_string()))?;
                let meta_json: String = row.get(10).map_err(|e| StoreError::Backend(e.to_string()))?;
                let extra_json: String = row.get(11).map_err(|e| StoreError::Backend(e.to_string()))?;
                let section_path: Option<Vec<String>> = serde_json::from_str(&section_path_json).ok();
                let meta: std::collections::BTreeMap<String, String> = serde_json::from_str(&meta_json).unwrap_or_default();
                let extra: std::collections::BTreeMap<String, JsonValue> = serde_json::from_str(&extra_json).unwrap_or_default();
                Some(ChunkRecord {
                    schema_version: schema_version as u16,
                    doc_id: DocumentId(doc_id2),
                    chunk_id: ChunkId(chunk_id),
                    source_uri,
                    source_mime,
                    extracted_at,
                    page_start: page_start_opt.and_then(|v| u32::try_from(v).ok()),
                    page_end: page_end_opt.and_then(|v| u32::try_from(v).ok()),
                    text,
                    section_path,
                    meta,
                    extra,
                })
            } else { None }
        };

        Ok((prev, next))
    }
}

