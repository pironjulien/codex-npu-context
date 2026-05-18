---
name: codex-npu-context
description: Use when a task would benefit from semantic retrieval over the user's local Codex context, prior Codex chats, indexed repositories, old decisions, errors, setup notes, or when exact filenames/keywords are unknown.
---

# Codex NPU Context

Use `codex_npu_search` when local semantic context is more useful than exact keyword search.

Good triggers:

- The user asks to retrieve prior decisions, old Codex chats, setup details, or "what did we do before?"
- The relevant file name or exact keyword is uncertain.
- A task spans multiple repos, local tooling, setup notes, debugging history, or Codex config.
- `rg` is likely to miss synonyms or conversational context.

Workflow:

1. Call `codex_npu_status` if you need to confirm the index exists.
2. Call `codex_npu_search` with a concise natural-language query.
3. Treat returned paths and excerpts as leads.
4. Read the real files before editing or making claims.
5. If `has_confident_result` is false, say the local index does not have a confident match and fall back to exact search or ask to rebuild the index.

Important:

- This is a retrieval aid, not a source of truth.
- Use `rg` for exact strings, symbols, and filenames.
- Rebuild the index after important new sessions, repo changes, or notes.
