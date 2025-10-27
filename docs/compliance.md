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
