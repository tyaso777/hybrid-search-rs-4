# Agent Guidance (Repository Root)

This file defines default operating principles for the Codex CLI agent when working in this repository. More deeply nested AGENTS.md files override this one. Direct user instructions in the chat always take precedence over this file.

Principles

- Write all code comments, user‑facing README text, and Git commit messages in English.
- Do not create Git commits or push changes without explicit user permission.
- Before starting an implementation, briefly confirm the implementation approach with the user (design, scope, affected files, validation plan).
- If a single user request appears large or ambiguous, propose a staged or partial implementation plan first, and proceed incrementally after the user confirms.
- Do not perform destructive actions (e.g., rm, reset, force‑push, mass rewrites) without prior explicit user agreement; prefer reversible steps and show a plan/dry‑run first.

Notes

- Prefer minimal, focused diffs that respect the existing code style.
- When in doubt about requirements or trade‑offs, ask clarifying questions before making changes.
