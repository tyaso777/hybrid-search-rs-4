#![allow(dead_code)]

// Real Tantivy-backed searcher is provided behind the `tantivy-impl` feature.
// The default build compiles a stub to keep the crate lightweight and portable.

#[cfg(feature = "tantivy-impl")]
pub use real::TantivyIndex;

#[cfg(not(feature = "tantivy-impl"))]
pub struct TantivyIndex;

#[cfg(not(feature = "tantivy-impl"))]
impl TantivyIndex {
    pub fn new_ram() -> Result<Self, ()> { Ok(Self) }
    pub fn upsert_records(&self, _records: &[chunk_model::ChunkRecord]) -> Result<(), ()> { Ok(()) }
}

#[cfg(not(feature = "tantivy-impl"))]
impl crate::TextSearcher for TantivyIndex {
    fn name(&self) -> &'static str { "tantivy" }
    fn caps(&self) -> crate::IndexCaps {
        crate::IndexCaps {
            can_prefilter_doc_id_eq: true,
            can_prefilter_doc_id_in: true,
            can_prefilter_source_prefix: true,
            can_prefilter_meta: false,
            can_prefilter_range_numeric: false,
            can_prefilter_range_date: true,
        }
    }
    fn search_ids(&self, _store: &dyn crate::ChunkStoreRead, _query: &str, _filters: &[crate::FilterClause], _opts: &crate::SearchOptions) -> Vec<crate::TextMatch> { Vec::new() }
}

#[cfg(feature = "tantivy-impl")]
mod real {
    use chunk_model::ChunkRecord;
    use chrono::DateTime;
    use tantivy::collector::TopDocs;
    use tantivy::query::QueryParser;
    use tantivy::schema::{Schema, TEXT, STRING, STORED, NumericOptions};
    use tantivy::Index;
    use tantivy::doc;
    use crate::{ChunkStoreRead, FilterClause, FilterOp, IndexCaps, SearchOptions, TextMatch, TextSearcher};

    pub struct TantivyIndex {
        schema: Schema,
        index: Index,
        reader: tantivy::IndexReader,
        // fields
        f_text: tantivy::schema::Field,
        f_chunk_id: tantivy::schema::Field,
        f_doc_id: tantivy::schema::Field,
        f_source_uri: tantivy::schema::Field,
        f_extracted_at: tantivy::schema::Field,
        f_extracted_at_ts: tantivy::schema::Field,
    }

    impl TantivyIndex {
        pub fn new_ram() -> tantivy::Result<Self> {
            let mut schema_builder = Schema::builder();
            let text = schema_builder.add_text_field("text", TEXT);
            let chunk_id = schema_builder.add_text_field("chunk_id", STRING | STORED);
            let doc_id = schema_builder.add_text_field("doc_id", STRING);
            let source_uri = schema_builder.add_text_field("source_uri", STRING);
            // Store as ISO-8601 string and prepare numeric epoch field for ranges
            let extracted_at = schema_builder.add_text_field("extracted_at", STRING);
            let num_opts = NumericOptions::default().set_fast().set_indexed();
            let extracted_at_ts = schema_builder.add_i64_field("extracted_at_ts", num_opts);
            let schema = schema_builder.build();
            let index = Index::create_in_ram(schema.clone());
            let reader = index.reader()?;
            Ok(Self { schema, index, reader, f_text: text, f_chunk_id: chunk_id, f_doc_id: doc_id, f_source_uri: source_uri, f_extracted_at: extracted_at, f_extracted_at_ts: extracted_at_ts })
        }

        pub fn upsert_records(&self, records: &[ChunkRecord]) -> tantivy::Result<()> {
            let mut writer = self.index.writer(50_000_000)?;
            for rec in records {
                let mut doc = tantivy::doc! {
                    self.f_chunk_id => rec.chunk_id.0.clone(),
                    self.f_doc_id => rec.doc_id.0.clone(),
                    self.f_source_uri => rec.source_uri.clone(),
                    self.f_extracted_at => rec.extracted_at.clone(),
                    self.f_text => rec.text.clone(),
                };
                if let Some(ts) = parse_rfc3339_to_ts(&rec.extracted_at) {
                    doc.add_i64(self.f_extracted_at_ts, ts);
                }
                let _ = writer.add_document(doc);
            }
            writer.commit()?;
            self.reader.reload()?;
            Ok(())
        }
    }

    impl TextSearcher for TantivyIndex {
        fn name(&self) -> &'static str { "tantivy" }
        fn caps(&self) -> IndexCaps { IndexCaps { can_prefilter_doc_id_eq: true, can_prefilter_doc_id_in: true, can_prefilter_source_prefix: true, can_prefilter_meta: false, can_prefilter_range_numeric: false, can_prefilter_range_date: true } }
        fn search_ids(&self, _store: &dyn ChunkStoreRead, query: &str, filters: &[FilterClause], opts: &SearchOptions) -> Vec<TextMatch> {
            if query.trim().is_empty() || opts.top_k == 0 { return Vec::new(); }
            let mut q = query.trim().to_string();
            let mut doc_parts: Vec<String> = Vec::new();
            for fc in filters { if let crate::FilterOp::DocIdEq(v) = &fc.op { doc_parts.push(format!("doc_id:\"{}\"", escape_q(v))); } }
            for fc in filters { if let crate::FilterOp::DocIdIn(vs) = &fc.op { for v in vs { doc_parts.push(format!("doc_id:\"{}\"", escape_q(v))); } } }
            if !doc_parts.is_empty() { q.push(' '); if doc_parts.len() > 1 { q.push('('); } q.push_str(&doc_parts.join(" OR ")); if doc_parts.len() > 1 { q.push(')'); } }
            for fc in filters { if let crate::FilterOp::SourceUriPrefix(p) = &fc.op { q.push_str(&format!(" source_uri:{}*", escape_term(p))); } }
            for fc in filters { if let crate::FilterOp::RangeIsoDate { key, start, end, .. } = &fc.op { if key == "extracted_at" { let lower = start.as_deref().and_then(parse_rfc3339_to_ts); let upper = end.as_deref().and_then(parse_rfc3339_to_ts); let mut part = String::from(" extracted_at_ts:"); part.push('['); match lower { Some(v) => part.push_str(&v.to_string()), None => part.push_str(&i64::MIN.to_string()) } part.push_str(" TO "); match upper { Some(v) => part.push_str(&v.to_string()), None => part.push_str(&i64::MAX.to_string()) } part.push(']'); q.push_str(&part); } } }
            let parser = QueryParser::for_index(&self.index, vec![self.f_text, self.f_doc_id, self.f_source_uri, self.f_extracted_at_ts]);
            let parsed = match parser.parse_query(&q) { Ok(q) => q, Err(_) => return Vec::new() };
            let searcher = self.reader.searcher();
            let fetch_n = (opts.top_k.saturating_mul(opts.fetch_factor)).max(opts.top_k);
            let top_docs = match searcher.search(&parsed, &TopDocs::with_limit(fetch_n)) { Ok(hits) => hits, Err(_) => return Vec::new() };
            let mut out = Vec::with_capacity(top_docs.len());
            for (raw_score, addr) in top_docs {
                if let Ok(doc) = searcher.doc::<tantivy::schema::document::TantivyDocument>(addr) {
                    if let Some(v) = doc.get_first(self.f_chunk_id) {
                        if let tantivy::schema::OwnedValue::Str(cid) = v {
                            let score = 1.0f32 / (1.0f32 + (-raw_score).exp());
                            out.push(TextMatch { chunk_id: chunk_model::ChunkId(cid.to_string()), score, raw_score });
                        }
                    }
                }
            }
            out
        }
    }

    fn escape_q(s: &str) -> String { s.replace('"', "\\\"") }
    fn escape_term(s: &str) -> String { s.replace(' ', "\\ ") }
    fn parse_rfc3339_to_ts(s: &str) -> Option<i64> { if s.is_empty() { None } else { DateTime::parse_from_rfc3339(s).ok().map(|dt| dt.timestamp()) } }
}

