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

    // Only embedded IPADIC
    println!("-- embedded://ipadic --");
    let dictionary = load_dictionary("embedded://ipadic").expect("load embedded ipadic");
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
