---
name: codex-npu-context
description: Use when a task would benefit from semantic retrieval over local Codex context, prior chats, indexed repositories, decisions, errors, setup notes, or when exact filenames and keywords are unknown. Requires the codex-npu-context MCP server.
---

# Codex NPU Context

Use the `codex_npu_search` MCP tool when local semantic context is more useful than exact keyword search.

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
5. If `status` is `no_confident_result` or `has_confident_result` is false, say the local index does not have a confident match and fall back to exact search or ask to rebuild the index.

Important:

- This is a retrieval aid, not a source of truth.
- Use `rg` for exact strings, symbols, and filenames.
- Rebuild the index after important new sessions, repo changes, or notes.
