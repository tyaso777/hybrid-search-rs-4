Test Data Layout

- public: small, redistributable samples that can be checked into Git.
- no-redistribute: local-only data (copyrighted, confidential, or very large). Ignored by Git.

Conventions
- Prefer placing shared samples under `public/`.
- Keep sensitive or large data under `no-redistribute/`.
- Tools may also honor `TESTDATA_DIR` to point at an external data root.

