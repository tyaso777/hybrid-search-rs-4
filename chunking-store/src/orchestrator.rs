use crate::sqlite_repo::SqliteRepo;
use crate::{FilterClause, TextIndexMaintainer, VectorIndexMaintainer, StoreError, ChunkPrimaryStore};
use chunk_model::{ChunkRecord, ChunkId};

#[derive(Debug, thiserror::Error)]
pub enum OrchestratorError {
    #[error("store error: {0}")]
    Store(#[from] StoreError),
    #[error("index error: {0}")]
    Index(String),
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DeleteReport {
    pub total_ids: usize,
    pub db_deleted: usize,
    pub text_delete_attempts: usize,
    pub vector_delete_attempts: usize,
    pub batches: usize,
}

/// Orchestrate deletion: list IDs by filters in batches, delete from DB, then delete from indexes.
///
/// Strategy: repeatedly fetch first N matching IDs (OFFSET 0) and delete; rows shift as we remove.
pub fn delete_by_filter_orchestrated(
    repo: &mut SqliteRepo,
    filters: &[FilterClause],
    batch_size: usize,
    text_indexes: &[&dyn TextIndexMaintainer],
    vector_indexes: &mut [&mut dyn VectorIndexMaintainer],
)
    -> Result<DeleteReport, OrchestratorError>
{
    let mut report = DeleteReport::default();
    let batch = batch_size.max(1);

    loop {
        let ids = repo.list_chunk_ids_by_filter(filters, batch, 0)?;
        if ids.is_empty() { break; }
        report.total_ids += ids.len();
        report.batches += 1;

        let n = repo.delete_by_ids(&ids)?;
        report.db_deleted += n;

        for ti in text_indexes {
            ti.delete_by_ids(&ids).map_err(|e| OrchestratorError::Index(format!("{e}")))?;
            report.text_delete_attempts += ids.len();
        }
        for vi in vector_indexes.iter_mut() {
            vi.delete_by_ids(&ids).map_err(|e| OrchestratorError::Index(format!("{e}")))?;
            report.vector_delete_attempts += ids.len();
        }
    }
    Ok(report)
}

/// Ingest orchestrator: upsert into DB, then update text/vector indexes.
pub fn ingest_chunks_orchestrated(
    repo: &mut SqliteRepo,
    records: &[ChunkRecord],
    text_indexes: &[&dyn TextIndexMaintainer],
    vector_indexes: &mut [&mut dyn VectorIndexMaintainer],
    vectors: Option<&[(ChunkId, Vec<f32>)]>,
) -> Result<(), OrchestratorError> {
    if records.is_empty() { return Ok(()); }
    repo.upsert_chunks(records.to_vec())?;
    // Ensure FTS5 is populated in rare cases where triggers lag at first creation
    let _ = repo.maybe_rebuild_fts();
    for ti in text_indexes {
        ti.upsert(records).map_err(|e| OrchestratorError::Index(format!("{e}")))?;
    }
    if let Some(v) = vectors {
        for vi in vector_indexes.iter_mut() {
            vi.upsert_vectors(v).map_err(|e| OrchestratorError::Index(format!("{e}")))?;
        }
    }
    Ok(())
}
