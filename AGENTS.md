# Agent Guidance (Repository Root)

This file defines default operating principles for the Codex CLI agent when working in this repository. More deeply nested AGENTS.md files override this one. Direct user instructions in the chat always take precedence over this file.

Principles

- Write all code comments, user‑facing README text, and Git commit messages in English.
- Never create Git commits or push changes unless the user explicitly instructs you to. After implementing changes, pause and ask the user whether to commit; do not commit by default.
- When you do create a commit (only after explicit instruction), always include the short commit hash (e.g., 265ac03) in your response so the user can reference it.
- Before starting an implementation, briefly confirm the implementation approach with the user (design, scope, affected files, validation plan).
- If a single user request appears large or ambiguous, propose a staged or partial implementation plan first, and proceed incrementally after the user confirms.
- Do not perform destructive actions (e.g., rm, reset, force‑push, mass rewrites) without prior explicit user agreement; prefer reversible steps and show a plan/dry‑run first.

Notes

- Prefer minimal, focused diffs that respect the existing code style.
- When in doubt about requirements or trade‑offs, ask clarifying questions before making changes.

Build/Validation

- After editing Rust code, run `cargo check` to validate builds quickly.
- For focused changes, run `cargo check -p <crate>` (e.g., `cargo check -p hybrid-service-gui`).
- If `cargo` isn’t available in the environment, skip and ask the user to run it locally.
- Optional: if formatting is configured, run `cargo fmt -- --check` after making code changes (before asking the user to commit), and only commit when the user approves.
