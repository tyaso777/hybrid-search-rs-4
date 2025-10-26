use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use chunk_model::ChunkId;
use hnsw_rs::prelude::*;

use crate::{ChunkStoreRead, FilterClause, SearchOptions, TextMatch, VectorSearcher};

/// HNSW-based vector index (Cosine distance). Persists by snapshotting vectors + id map.
pub struct HnswIndex {
    dim: usize,
    hnsw: Hnsw<'static, f32, DistCosine>,
    /// Map chunk_id -> internal label
    id_map: HashMap<String, usize>,
    /// Reverse map internal label -> chunk_id
    rev_map: Vec<String>,
    /// Stored vectors for persistence and rebuild
    vectors: Vec<Vec<f32>>,
    /// Tombstoned labels (deleted)
    tombstones: HashSet<usize>,
}

impl HnswIndex {
    pub fn new(dim: usize, expected: usize) -> Self {
        let max_nb_conn = 16;
        let ef_c = 200;
        let num_layers = 16;
        let hnsw = Hnsw::<f32, DistCosine>::new(max_nb_conn, expected, num_layers, ef_c, DistCosine {});
        Self { dim, hnsw, id_map: HashMap::new(), rev_map: Vec::new(), vectors: Vec::new(), tombstones: HashSet::new() }
    }

    /// Upsert vectors; duplicate chunk_id replaces previous vector by reinsert (no true delete in HNSW).
    pub fn upsert(&mut self, items: &[(ChunkId, Vec<f32>)]) {
        for (cid, v) in items {
            if v.len() != self.dim { continue; }
            let label = if let Some(&lbl) = self.id_map.get(&cid.0) {
                // naive: just insert again; HNSW has no delete. Rebuild recommended for heavy churn.
                lbl
            } else {
                let lbl = self.rev_map.len();
                self.id_map.insert(cid.0.clone(), lbl);
                self.rev_map.push(cid.0.clone());
                self.vectors.push(v.clone());
                lbl
            };
            let _ = self.hnsw.insert((&v[..], label));
        }
        // optional dump
    }

    /// Snapshot vectors + map to a directory (rebuilds index on load).
    pub fn save<P: AsRef<Path>>(&self, dir: P) -> std::io::Result<()> {
        let dir = dir.as_ref();
        fs::create_dir_all(dir)?;
        let map_path = dir.join("map.tsv.tmp");
        let vec_path = dir.join("vectors.bin.tmp");
        {
            let mut w = fs::File::create(&map_path)?;
            for (i, cid) in self.rev_map.iter().enumerate() {
                use std::io::Write;
                writeln!(w, "{i}\t{cid}")?;
            }
        }
        {
            let mut w = fs::File::create(&vec_path)?;
            use std::io::Write;
            // binary: [u32 dim][f32..] repeated
            for v in &self.vectors {
                let dim = v.len() as u32;
                w.write_all(&dim.to_le_bytes())?;
                let bytes: &[u8] = bytemuck::cast_slice(&v[..]);
                w.write_all(bytes)?;
            }
        }
        fs::rename(map_path, dir.join("map.tsv"))?;
        fs::rename(vec_path, dir.join("vectors.bin"))?;
        Ok(())
    }

    /// Load snapshot and rebuild HNSW.
    pub fn load<P: AsRef<Path>>(dir: P, dim: usize) -> std::io::Result<Self> {
        let dir = dir.as_ref();
        let map_txt = fs::read_to_string(dir.join("map.tsv"))?;
        let mut rev_map: Vec<String> = Vec::new();
        for line in map_txt.lines() {
            let mut it = line.splitn(2, '\t');
            let _idx = it.next();
            if let Some(cid) = it.next() { rev_map.push(cid.to_string()); }
        }
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(rev_map.len());
        let mut r = std::io::BufReader::new(fs::File::open(dir.join("vectors.bin"))?);
        use std::io::Read;
        loop {
            let mut len_buf = [0u8; 4];
            if let Err(_) = r.read_exact(&mut len_buf) { break; }
            let l = u32::from_le_bytes(len_buf) as usize;
            let mut vbytes = vec![0u8; 4 * l];
            r.read_exact(&mut vbytes)?;
            let vf32: Vec<f32> = bytemuck::cast_slice(&vbytes).to_vec();
            vectors.push(vf32);
        }
        let expected = vectors.len().max(1000);
        let hnsw = Hnsw::<f32, DistCosine>::new(16, expected, 16, 200, DistCosine {});
        let mut id_map = HashMap::new();
        for (i, v) in vectors.iter().enumerate() {
            id_map.insert(rev_map[i].clone(), i);
            let _ = hnsw.insert((&v[..], i));
        }
        let this = Self { dim, hnsw, id_map, rev_map, vectors, tombstones: HashSet::new() };
        Ok(this)
    }
}

impl VectorSearcher for HnswIndex {
    fn name(&self) -> &'static str { "hnsw" }
    fn dimension(&self) -> usize { self.dim }

    fn knn_ids(
        &self,
        _store: &dyn ChunkStoreRead,
        query: &[f32],
        _filters: &[FilterClause],
        opts: &SearchOptions,
    ) -> Vec<TextMatch> {
        if query.len() != self.dim || opts.top_k == 0 { return Vec::new(); }
        let ef_s = (opts.top_k.saturating_mul(opts.fetch_factor)).max(opts.top_k);
        let knn = self.hnsw.search(query, opts.top_k * 5, ef_s);
        let mut out = Vec::new();
        // No prefilter; apply post-filter on metadata if needed (needs external store if required)
        for el in knn {
            let label = el.d_id;
            if self.tombstones.contains(&label) { continue; }
            let cid = &self.rev_map[label];
            let dist = el.distance; // cosine distance (smaller is better)
            let score = 1.0f32 - (dist as f32); // similarity approx
            out.push(TextMatch { chunk_id: ChunkId(cid.clone()), score, raw_score: dist as f32 });
            if out.len() >= opts.top_k { break; }
        }
        out
    }
}

impl crate::VectorIndexMaintainer for HnswIndex {
    fn upsert_vectors(&mut self, items: &[(chunk_model::ChunkId, Vec<f32>)]) -> Result<(), crate::IndexError> {
        self.upsert(items);
        Ok(())
    }

    fn delete_by_ids(&mut self, ids: &[chunk_model::ChunkId]) -> Result<(), crate::IndexError> {
        for cid in ids {
            if let Some(&lbl) = self.id_map.get(&cid.0) {
                self.tombstones.insert(lbl);
            }
        }
        Ok(())
    }
}

