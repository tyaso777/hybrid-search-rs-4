# Compliance and Licensing

Back to the high-level overview in README: [../README.md#compliance--security](../README.md#compliance--security)

This document summarizes how to generate and package third‑party license information and how we check for known vulnerabilities.

## Project License

- This project is licensed under MIT. See `LICENSE`.

## Third‑Party Notices

Generated artifacts placed under `reports/`:

- `THIRD-PARTY-NOTICES.txt`
  - Summarized list of third‑party crates with license identifiers and repository URLs (workspace crates excluded)
- `THIRD-PARTY-LICENSES.txt`
  - Consolidated license texts per license, with a list of crates using each license (generated via cargo-about)

When distributing binaries or packaged apps, include both of the above and `LICENSE`.

## Generating Reports

Windows (PowerShell):

```
powershell -ExecutionPolicy Bypass -File scripts/generate_reports.ps1
```

Bash:

```
bash scripts/generate_reports.sh
```

Outputs:

- Vulnerabilities (RustSec): `reports/cargo-audit.txt`, `reports/cargo-audit.json`
- Licenses summary: `reports/license.txt`, `reports/license.json`
- Third‑party notices: `reports/THIRD-PARTY-NOTICES.txt`
- License texts (cargo-about): `reports/THIRD-PARTY-LICENSES.txt` and `reports/about.json`

## Policy and CI

- Policy file: `deny.toml`
- CI workflow: `.github/workflows/cargo-deny.yml`
- Local run: `cargo deny check`

Notes:

- Some upstream crates may be flagged as unmaintained (no safe upgrade available). We track such advisories in `deny.toml` under `advisories.ignore` with reasons and review them periodically.
- Font license references (e.g., `LicenseRef-UFL-1.0`) are handled via exceptions in `deny.toml` where necessary.

## Release Procedure

Follow this procedure when preparing a distributable (ZIP/installer/binary):

1) Generate fresh reports
   - Windows: `powershell -ExecutionPolicy Bypass -File scripts/generate_reports.ps1`
   - Bash: `bash scripts/generate_reports.sh`
   - Verify outputs under `reports/` are up to date.

2) Run policy checks locally
   - `cargo deny check`
   - Address issues or document temporary exceptions in `deny.toml` with a reason.

3) Prepare distribution payload
   - Include files:
     - `LICENSE` (project license: MIT)
     - `reports/THIRD-PARTY-NOTICES.txt` (third‑party summary)
     - `reports/THIRD-PARTY-LICENSES.txt` (full license texts)
   - If bundling ONNX Runtime (ORT):
     - Include ORT’s MIT license file (typically `LICENSE` in the ORT archive)
     - Include ORT’s `ThirdPartyNotices` if provided in the ORT archive
   - If bundling other runtime/providers (CUDA, TensorRT, DirectML, OpenVINO, etc.):
     - Verify and include each vendor’s required license/notice files.

4) Non‑code assets (must review and include licenses as applicable)
   - Model files (ONNX): ensure the model’s own license permits redistribution; include its license text
   - Dictionaries (e.g., Lindera embedded IPADIC): include the relevant license(s) (e.g., BSD‑3‑Clause) if redistributed
   - Fonts: include font licenses (e.g., OFL‑1.1, UFL) and follow name change/notice requirements when applicable

5) Release notes and artifacts
   - Summarize notable dependency/license changes
   - Attach the built artifacts and ensure required license/notice files are present
