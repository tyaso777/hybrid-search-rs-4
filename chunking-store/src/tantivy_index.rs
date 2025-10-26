#![allow(dead_code)]

// Real Tantivy-backed searcher is provided behind the `tantivy-impl` feature.
// The default build compiles a stub to keep the crate lightweight and portable.

#[cfg(feature = "tantivy-impl")]
pub use real::{TantivyIndex, TokenCombine};

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
    use tantivy::query::{BooleanQuery, Occur, QueryParser, RangeQuery, TermQuery};
    use tantivy::schema::{IndexRecordOption, NumericOptions, Schema, STRING, STORED, TextFieldIndexing, TextOptions};
    use tantivy::schema::Value as _;
    use tantivy::{Index, Term};
    use tantivy::doc;
    use tantivy::tokenizer::TokenStream;
    use crate::{ChunkStoreRead, FilterClause, FilterOp, IndexCaps, SearchOptions, TextMatch, TextSearcher};
    // use std::ops::Range;
    use std::path::Path;

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

    #[derive(Debug, Clone, Copy)]
    pub enum TokenCombine { AND, OR }

    impl TantivyIndex {
        fn build_schema() -> (Schema, tantivy::schema::Field, tantivy::schema::Field, tantivy::schema::Field, tantivy::schema::Field, tantivy::schema::Field, tantivy::schema::Field) {
            let mut schema_builder = Schema::builder();
            let mut text_indexing = TextFieldIndexing::default();
            text_indexing = text_indexing.set_tokenizer("ja");
            text_indexing = text_indexing.set_index_option(IndexRecordOption::WithFreqsAndPositions);
            let text_options = TextOptions::default().set_indexing_options(text_indexing);
            let text = schema_builder.add_text_field("text", text_options);
            let chunk_id = schema_builder.add_text_field("chunk_id", STRING | STORED);
            let doc_id = schema_builder.add_text_field("doc_id", STRING);
            let source_uri = schema_builder.add_text_field("source_uri", STRING);
            let extracted_at = schema_builder.add_text_field("extracted_at", STRING);
            let num_opts = NumericOptions::default().set_fast().set_indexed();
            let extracted_at_ts = schema_builder.add_i64_field("extracted_at_ts", num_opts);
            let schema = schema_builder.build();
            (schema, text, chunk_id, doc_id, source_uri, extracted_at, extracted_at_ts)
        }

        fn register_ja_tokenizer(index: &Index) {
            use lindera::dictionary::load_dictionary;
            use lindera::mode::Mode;
            use lindera::segmenter::Segmenter;
            use lindera_tantivy::tokenizer::LinderaTokenizer;
            let dictionary = load_dictionary("embedded://ipadic").expect("load embedded ipadic");
            let user_dictionary = None;
            let segmenter = Segmenter::new(Mode::Normal, dictionary, user_dictionary);
            let tokenizer = LinderaTokenizer::from_segmenter(segmenter);
            index.tokenizers().register("ja", tokenizer);
        }

        pub fn new_ram() -> tantivy::Result<Self> {
            let (schema, text, chunk_id, doc_id, source_uri, extracted_at, extracted_at_ts) = Self::build_schema();
            let index = Index::create_in_ram(schema.clone());
            Self::register_ja_tokenizer(&index);
            let reader = index.reader()?;
            Ok(Self { schema, index, reader, f_text: text, f_chunk_id: chunk_id, f_doc_id: doc_id, f_source_uri: source_uri, f_extracted_at: extracted_at, f_extracted_at_ts: extracted_at_ts })
        }

        /// Open an existing on-disk index at `path`, or create a new one if absent.
        pub fn open_or_create_dir<P: AsRef<Path>>(path: P) -> tantivy::Result<Self> {
            let dir = path.as_ref();
            std::fs::create_dir_all(dir).map_err(|e| tantivy::TantivyError::IoError(e.into()))?;
            let index = match Index::open_in_dir(dir) {
                Ok(idx) => idx,
                Err(_) => {
                    let (schema, text, chunk_id, doc_id, source_uri, extracted_at, extracted_at_ts) = Self::build_schema();
                    let idx = Index::create_in_dir(dir, schema.clone())?;
                    Self::register_ja_tokenizer(&idx);
                    let reader = idx.reader()?;
                    return Ok(Self { schema, index: idx, reader, f_text: text, f_chunk_id: chunk_id, f_doc_id: doc_id, f_source_uri: source_uri, f_extracted_at: extracted_at, f_extracted_at_ts: extracted_at_ts });
                }
            };
            // existing index: derive fields by name
            let schema = index.schema();
            let f_text = schema.get_field("text")?;
            let f_chunk_id = schema.get_field("chunk_id")?;
            let f_doc_id = schema.get_field("doc_id")?;
            let f_source_uri = schema.get_field("source_uri")?;
            let f_extracted_at = schema.get_field("extracted_at")?;
            let f_extracted_at_ts = schema.get_field("extracted_at_ts")?;
            Self::register_ja_tokenizer(&index);
            let reader = index.reader()?;
            Ok(Self { schema, index, reader, f_text, f_chunk_id, f_doc_id, f_source_uri, f_extracted_at, f_extracted_at_ts })
        }

        pub fn upsert_records(&self, records: &[ChunkRecord]) -> tantivy::Result<()> {
            let mut writer = self.index.writer(50_000_000)?;
            for rec in records {
                // emulate UPSERT: delete existing doc by chunk_id then add
                let term = Term::from_field_text(self.f_chunk_id, &rec.chunk_id.0);
                writer.delete_term(term);

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

        /// Build a query by tokenizing the input with the field analyzer and
        /// combining terms via AND/OR (avoids overly strict phrase matching).
        pub fn search_ids_tokenized(
            &self,
            _store: &dyn ChunkStoreRead,
            query: &str,
            filters: &[FilterClause],
            opts: &SearchOptions,
            combine: TokenCombine,
        ) -> Vec<TextMatch> {
            if query.trim().is_empty() || opts.top_k == 0 { return Vec::new(); }

            // 1) Tokenize the query using the field analyzer for `text`.
            let mut toks: Vec<String> = Vec::new();
            if let Ok(mut analyzer) = self.index.tokenizer_for_field(self.f_text) {
                let mut ts = analyzer.token_stream(query);
                while ts.advance() {
                    let t = ts.token();
                    if !t.text.is_empty() { toks.push(t.text.clone()); }
                }
            }
            if toks.is_empty() { return Vec::new(); }

            // 2) Build boolean of terms.
            let mut clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
            for tk in &toks {
                let term = Term::from_field_text(self.f_text, tk);
                let tq = TermQuery::new(term, IndexRecordOption::Basic);
                clauses.push((match combine { TokenCombine::AND => Occur::Must, TokenCombine::OR => Occur::Should }, Box::new(tq)));
            }

            // 3) Append filters (same as default implementation)
            // doc_id eq/in
            let mut doc_terms: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
            for fc in filters {
                match &fc.op {
                    FilterOp::DocIdEq(v) => {
                        let term = Term::from_field_text(self.f_doc_id, v);
                        doc_terms.push((Occur::Should, Box::new(TermQuery::new(term, IndexRecordOption::Basic))));
                    }
                    FilterOp::DocIdIn(vs) => {
                        for v in vs {
                            let term = Term::from_field_text(self.f_doc_id, v);
                            doc_terms.push((Occur::Should, Box::new(TermQuery::new(term, IndexRecordOption::Basic))));
                        }
                    }
                    _ => {}
                }
            }
            if !doc_terms.is_empty() {
                clauses.push((Occur::Must, Box::new(BooleanQuery::from(doc_terms))));
            }

            // source_uri prefix
            for fc in filters {
                if let FilterOp::SourceUriPrefix(p) = &fc.op {
                    let uri_parser = QueryParser::for_index(&self.index, vec![self.f_source_uri]);
                    let qstr = format!("{}*", escape_term(p));
                    if let Ok(q) = uri_parser.parse_query(&qstr) { clauses.push((Occur::Must, q)); }
                }
            }
            // extracted_at range
            for fc in filters {
                if let FilterOp::RangeIsoDate { key, start, end, start_incl, end_incl } = &fc.op {
                    if key == "extracted_at" {
                        use std::ops::Bound;
                        let lower = match start.as_deref().and_then(parse_rfc3339_to_ts) {
                            Some(s) => Bound::Included(Term::from_field_i64(self.f_extracted_at_ts, if *start_incl { s } else { s.saturating_add(1) })),
                            None => Bound::Unbounded,
                        };
                        let upper = match end.as_deref().and_then(parse_rfc3339_to_ts) {
                            Some(e) => Bound::Included(Term::from_field_i64(self.f_extracted_at_ts, if *end_incl { e } else { e.saturating_sub(1) })),
                            None => Bound::Unbounded,
                        };
                        clauses.push((Occur::Must, Box::new(RangeQuery::new(lower, upper))));
                    }
                }
            }

            // 4) Execute
            let combined = BooleanQuery::from(clauses);
            let searcher = self.reader.searcher();
            let fetch_n = (opts.top_k.saturating_mul(opts.fetch_factor)).max(opts.top_k);
            let top_docs = match searcher.search(&combined, &TopDocs::with_limit(fetch_n)) { Ok(hits) => hits, Err(_) => return Vec::new() };
            let mut out = Vec::with_capacity(top_docs.len());
            for (raw_score, addr) in top_docs {
                if let Ok(doc) = searcher.doc::<tantivy::schema::document::TantivyDocument>(addr) {
                    if let Some(v) = doc.get_first(self.f_chunk_id) {
                        if let Some(cid) = v.as_str() {
                            let score = 1.0f32 / (1.0f32 + (-raw_score).exp());
                            out.push(TextMatch { chunk_id: chunk_model::ChunkId(cid.to_string()), score, raw_score });
                        }
                    }
                }
            }
            out
        }
    }

    impl TextSearcher for TantivyIndex {
        fn name(&self) -> &'static str { "tantivy" }
        fn caps(&self) -> IndexCaps { IndexCaps { can_prefilter_doc_id_eq: true, can_prefilter_doc_id_in: true, can_prefilter_source_prefix: true, can_prefilter_meta: false, can_prefilter_range_numeric: false, can_prefilter_range_date: true } }
        fn search_ids(&self, _store: &dyn ChunkStoreRead, query: &str, filters: &[FilterClause], opts: &SearchOptions) -> Vec<TextMatch> {
            if query.trim().is_empty() || opts.top_k == 0 { return Vec::new(); }

            let mut clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();

            // Text part on text field only
            let text_parser = QueryParser::for_index(&self.index, vec![self.f_text]);
            let text_q = match text_parser.parse_query(query) { Ok(q) => q, Err(_) => return Vec::new() };
            clauses.push((Occur::Must, text_q));

            // doc_id eq/in
            let mut doc_terms: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
            for fc in filters {
                match &fc.op {
                    FilterOp::DocIdEq(v) => {
                        let term = Term::from_field_text(self.f_doc_id, v);
                        doc_terms.push((Occur::Should, Box::new(TermQuery::new(term, IndexRecordOption::Basic))));
                    }
                    FilterOp::DocIdIn(vs) => {
                        for v in vs {
                            let term = Term::from_field_text(self.f_doc_id, v);
                            doc_terms.push((Occur::Should, Box::new(TermQuery::new(term, IndexRecordOption::Basic))));
                        }
                    }
                    _ => {}
                }
            }
            if !doc_terms.is_empty() {
                clauses.push((Occur::Must, Box::new(BooleanQuery::from(doc_terms))));
            }

            // source_uri prefix via QueryParser on source_uri field with wildcard
            for fc in filters {
                if let FilterOp::SourceUriPrefix(p) = &fc.op {
                    let uri_parser = QueryParser::for_index(&self.index, vec![self.f_source_uri]);
                    let qstr = format!("{}*", escape_term(p));
                    if let Ok(q) = uri_parser.parse_query(&qstr) {
                        clauses.push((Occur::Must, q));
                    }
                }
            }

            // extracted_at range using numeric epoch fast field
            for fc in filters {
                if let FilterOp::RangeIsoDate { key, start, end, start_incl, end_incl } = &fc.op {
                    if key == "extracted_at" {
                        use std::ops::Bound;
                        let lower_bound = match start.as_deref().and_then(parse_rfc3339_to_ts) {
                            Some(s) => {
                                let v = if *start_incl { s } else { s.saturating_add(1) };
                                Bound::Included(Term::from_field_i64(self.f_extracted_at_ts, v))
                            }
                            None => Bound::Unbounded,
                        };
                        let upper_bound = match end.as_deref().and_then(parse_rfc3339_to_ts) {
                            Some(e) => {
                                let v = if *end_incl { e } else { e.saturating_sub(1) };
                                Bound::Included(Term::from_field_i64(self.f_extracted_at_ts, v))
                            }
                            None => Bound::Unbounded,
                        };
                        let rq = RangeQuery::new(lower_bound, upper_bound);
                        clauses.push((Occur::Must, Box::new(rq)));
                    }
                }
            }

            let combined = BooleanQuery::from(clauses);
            let searcher = self.reader.searcher();
            let fetch_n = (opts.top_k.saturating_mul(opts.fetch_factor)).max(opts.top_k);
            let top_docs = match searcher.search(&combined, &TopDocs::with_limit(fetch_n)) { Ok(hits) => hits, Err(_) => return Vec::new() };
            let mut out = Vec::with_capacity(top_docs.len());
            for (raw_score, addr) in top_docs {
                if let Ok(doc) = searcher.doc::<tantivy::schema::document::TantivyDocument>(addr) {
                    if let Some(v) = doc.get_first(self.f_chunk_id) {
                        if let Some(cid) = v.as_str() {
                            let score = 1.0f32 / (1.0f32 + (-raw_score).exp());
                            out.push(TextMatch { chunk_id: chunk_model::ChunkId(cid.to_string()), score, raw_score });
                        }
                    }
                }
            }
            out
        }
    }

    impl crate::TextIndexMaintainer for TantivyIndex {
        fn upsert(&self, records: &[chunk_model::ChunkRecord]) -> Result<(), crate::IndexError> {
            self.upsert_records(records).map_err(|e| crate::IndexError::Backend(e.to_string()))
        }

        fn delete_by_ids(&self, ids: &[chunk_model::ChunkId]) -> Result<(), crate::IndexError> {
            let mut writer = self.index.writer::<tantivy::schema::document::TantivyDocument>(50_000_000)
                .map_err(|e| crate::IndexError::Backend(e.to_string()))?;
            for cid in ids { let term = tantivy::Term::from_field_text(self.f_chunk_id, &cid.0); writer.delete_term(term); }
            writer.commit().map_err(|e| crate::IndexError::Backend(e.to_string()))?;
            self.reader.reload().map_err(|e| crate::IndexError::Backend(e.to_string()))?;
            Ok(())
        }

        fn delete_by_doc_ids(&self, doc_ids: &[String]) -> Result<(), crate::IndexError> {
            let mut writer = self.index.writer::<tantivy::schema::document::TantivyDocument>(50_000_000)
                .map_err(|e| crate::IndexError::Backend(e.to_string()))?;
            for did in doc_ids { let term = tantivy::Term::from_field_text(self.f_doc_id, did); writer.delete_term(term); }
            writer.commit().map_err(|e| crate::IndexError::Backend(e.to_string()))?;
            self.reader.reload().map_err(|e| crate::IndexError::Backend(e.to_string()))?;
            Ok(())
        }
    }

    fn escape_q(s: &str) -> String { s.replace('"', "\\\"") }
    fn escape_term(s: &str) -> String { s.replace(' ', "\\ ") }
    fn parse_rfc3339_to_ts(s: &str) -> Option<i64> { if s.is_empty() { None } else { DateTime::parse_from_rfc3339(s).ok().map(|dt| dt.timestamp()) } }
}

