This folder hosts local ONNX models for the embedding provider.

[← Back to embedding provider guide](../README.md)
[← Back to workspace README](../../README.md)

Place exported models under a subfolder, for example:

  ruri-v3-onnx/
    model.onnx
    tokenizer.json
    tokenizer.model (if any)
    tokenizer_config.json (optional)

Files here are ignored by Git except for this README and `.gitkeep` placeholders,
so large model files are never committed.

For full setup instructions (model export to ONNX, tokenizer/DLL locations, CLI/GUI usage), see the embedding provider guide:
- ../README.md
