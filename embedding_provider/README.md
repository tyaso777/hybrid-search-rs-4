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

- Model files (export or obtain as described in [Creating an ONNX Model](#creating-an-onnx-model))
  - `models/ruri-v3-onnx/model.onnx`
  - `models/ruri-v3-onnx/tokenizer.json` (and optional `tokenizer.model`, `tokenizer_config.json`)
- ONNX Runtime DLL (Windows CPU build example)
  - `bin/onnxruntime-win-x64-1.23.1/lib/onnxruntime.dll`
  - Download from the official ONNX Runtime releases: https://github.com/microsoft/onnxruntime/releases
    - Windows x64 (CPU): download the `onnxruntime-win-x64-<version>.zip`, then place `onnxruntime.dll` under the path above (or adjust in `src/config.rs`).
    - Linux/macOS: use the platform package and place `libonnxruntime.so` / `libonnxruntime.dylib` under a similar `bin/<platform>/lib/` path; update the configured path if needed.

Defaults are defined in `src/config.rs` and resolved relative to this crate, so both workspace-root or crate-root execution works.

To customize paths/dimensions globally, edit:
- `embedding_provider/src/config.rs:18` (`ONNX_STDIO_DEFAULTS`)

See also:
- [models/README.md](models/README.md) — where to place exported ONNX models and tokenizer files.
- [bin/README.md](bin/README.md) — where to place ONNX Runtime distributions (DLLs).

---

## Creating an ONNX Model

Export a Hugging Face encoder model to ONNX with Optimum.

- Official guide (Hugging Face Optimum ONNX):
  - https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/export_a_model

Example steps:

1) Environment
```
python -m venv .venv
. .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install optimum[onnxruntime] transformers torch
```

2) Export (optimum‑cli)

Generic (replace the model id):
```
optimum-cli export onnx \
  --model <huggingface-model-id> \
  --task feature-extraction \
  ./onnx-model
```

Specific example (cl-nagoya/ruri-v3-310m):
```
optimum-cli export onnx \
  --model cl-nagoya/ruri-v3-310m \
  --task feature-extraction \
  ./onnx-model
```

- For sentence‑embedding models, export the encoder and ensure the model outputs token hidden states.
- Copy `model.onnx` and tokenizer assets (`tokenizer.json`, and optionally `tokenizer.model`, `tokenizer_config.json`) into `models/<your-name>/`.

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

### Model loading modes

- `OnnxStdIoConfig { preload_model_to_memory: false }` (default)
  - Session initializes by opening the ONNX file (`commit_from_file`).
- `preload_model_to_memory: true`
  - Reads the model file fully into memory first and initializes from memory (`commit_from_memory`).
  - Useful when the model resides on a slow/latent network share; increases peak memory by roughly the model size during initialization.

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
