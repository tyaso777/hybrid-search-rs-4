use lindera::dictionary::load_dictionary;
use lindera::mode::Mode;
use lindera::segmenter::Segmenter;
use lindera_tantivy::tokenizer::LinderaTokenizer;
use tantivy::tokenizer::{Tokenizer, TokenStream};

fn main() {
    let samples = [
        "今日は令和6年です。",
        "新元号は令和です。",
        "令和",
        "推し活が楽しい",
        "生成AIを使う",
        "スプラトゥーン3を遊ぶ",
        "インボイス制度について",
        "ChatGPTを使っています",
    ];

    // Allow override via LINDERA_DICT_URI or LINDERA_DICT_DIR; fallback to embedded
    let dict_uri = dict_uri_from_env();
    println!("-- {} --", dict_uri);
    let dictionary = load_dictionary(&dict_uri).expect("load lindera dictionary");
    let user_dictionary = None;
    let mode = Mode::Normal;
    let segmenter = Segmenter::new(mode, dictionary, user_dictionary);
    let mut tokenizer = LinderaTokenizer::from_segmenter(segmenter);

    for s in samples.iter() {
        let mut stream = tokenizer.token_stream(s);
        let mut toks: Vec<String> = Vec::new();
        while stream.advance() {
            toks.push(stream.token().text.clone());
        }
        println!("{} => {}", s, toks.join(" | "));
    }
}

fn dict_uri_from_env() -> String {
    if let Ok(uri) = std::env::var("LINDERA_DICT_URI") { return uri; }
    if let Ok(dir) = std::env::var("LINDERA_DICT_DIR") {
        let p = std::path::PathBuf::from(dir);
        let abs = if p.is_absolute() { p } else { std::env::current_dir().unwrap().join(p) };
        let mut s = abs.to_string_lossy().replace('\\', "/");
        if !s.starts_with('/') { s = format!("/{s}"); }
        return format!("file://{s}");
    }
    // Default to embedded when available
    "embedded://ipadic".to_string()
}
