use hybrid_service::{HybridService, ServiceConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: cargo run -p hybrid-service --example ingest_and_search -- <FILE> <QUERY>");
        std::process::exit(1);
    }
    let file = &args[1];
    let query = &args[2];

    let cfg = ServiceConfig::default();
    let svc = HybridService::new(cfg)?;
    svc.ingest_file(file, None)?;

    let hits = svc.search_hybrid(query, 10, &[], 0.5, 0.5)?;
    println!("Results: {}", hits.len());
    for (i, h) in hits.iter().enumerate() {
        let preview: String = h.chunk.text.chars().take(80).collect();
        println!("{:>2}. [{}] {:.4} {}", i+1, h.chunk.chunk_id.0, h.score, preview);
    }
    Ok(())
}

