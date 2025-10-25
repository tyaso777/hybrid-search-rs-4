## Embedding Provider (ONNX Runtime)

[← Back to workspace README](../README.md)

This crate provides a local ONNX-based text embedding implementation using ONNX Runtime and Hugging Face tokenizers. It loads a tokenization model, runs an encoder ONNX model, and mean-pools token hidden states (masked) to produce sentence/document vectors.

### What You Get
- API: `embedder::{Embedder, OnnxStdIoEmbedder, OnnxStdIoConfig}`
- Defaults: `config::{ONNX_STDIO_DEFAULTS, default_stdio_config()}`
- CLI: `src/bin/embed_cli.rs`
- Tests: deterministic output, batch consistency, max-length enforcement

---

## Setup

Place the following under this crate directory (paths are relative to `embedding_provider/`):

- Model files
  - `models/ruri-v3-onnx/model.onnx`
  - `models/ruri-v3-onnx/tokenizer.json` (and optional `tokenizer.model`, `tokenizer_config.json`)
- ONNX Runtime DLL (Windows CPU build example)
  - `bin/onnxruntime-win-x64-1.23.1/lib/onnxruntime.dll`

Defaults are defined in `src/config.rs` and resolved relative to this crate, so both workspace-root or crate-root execution works.

To customize paths/dimensions globally, edit:
- `embedding_provider/src/config.rs:18` (`ONNX_STDIO_DEFAULTS`)

---

## Creating an ONNX Model

Export a Hugging Face encoder model to ONNX with Optimum. Example steps:

1) Environment
```
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install optimum[onnxruntime] transformers torch
```

2) Export
```
python -m optimum.exporters.onnx --model <huggingface-model-id> ./onnx-model
```
- For sentence-embedding models, ensure the exported graph is the encoder and outputs token hidden states.
- Copy `model.onnx` and tokenizer assets (`tokenizer.json`, `tokenizer.model` if any) into `models/<your-name>/`.

3) Optional quick validation (Python)
```
python - <<'PY'
import onnxruntime as ort, numpy as np
s = ort.InferenceSession("onnx-model/model.onnx")
name = s.get_inputs()[0].name
print(s.run(None, {name: np.zeros((1, 8), dtype=np.int64)}))
PY
```

---

## How Vectors Are Produced

- Input: `input_ids` and `attention_mask` from `tokenizers` (requires a `<pad>` token)
- Model output: rank-3 tensor `[batch, seq_len, hidden]`
- Pooling: attention mask–based mean pooling → `[batch, hidden]`
- Dimension check: pooled vector length must equal configured `dimension` (default 768)

Notes:
- No L2-normalization is applied by default; cosine/inner-product search may benefit from it (can be added if desired).
- ONNX Runtime is initialized once per process (cannot switch DLL after init).

References
- Implementation: `embedding_provider/src/embedder/mod.rs`
- Defaults: `embedding_provider/src/config.rs`

---

## CLI Usage

Single-run sanity check:
```
cargo run -p embedding-provider --bin embed_cli "your text"
```
Outputs the model path, runtime DLL path, input, vector length, and a short preview of values.

---

## GUI Demo (Desktop)

Interactive embedding with file pickers and formatted previews:
```
cargo run -p embedder-demo
```
Features
- Configure model ONNX, tokenizer JSON, ONNX Runtime DLL, dimension, and token limit
- Initialize model asynchronously (non-blocking); cancelable
- Text embedding preview (formatted 8-per-row) and full vector view
- Excel/CSV batch embedding (first sheet/first column or first column), optional output path auto-generation
- Japanese font fallback for proper CJK rendering; override via `EMBEDDER_DEMO_FONT`

---

## Tests

```
cargo test -p embedding-provider
```
Verifies deterministic output, batch consistency, and length limits. Requires the model/tokenizer/DLL in the default locations (or adjust defaults).
