# Security Policy

See README for a high-level overview of checks and scripts: [README.md#compliance--security](README.md#compliance--security)

## Supported Versions

We maintain security checks continuously via CI (cargo-deny) on the default branch. Releases or tags are assessed on a best‑effort basis.

## Reporting a Vulnerability

- Please open a private security advisory or contact maintainers via a private channel. If that’s not possible, open a minimal public issue without sensitive details and request a secure follow‑up.
- Include:
  - Affected crate(s) and version(s)
  - Reproduction steps or proof‑of‑concept
  - Impact and suggested mitigation if known

## Dependency Vulnerabilities

We use the following to monitor dependencies:

- `cargo deny check` in CI (see `.github/workflows/cargo-deny.yml`)
- Local generation scripts (`scripts/generate_reports.*`) for RustSec and license reports

If you find issues in transitive dependencies, feel free to open an issue linking to the upstream advisory.
