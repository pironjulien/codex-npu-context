---
name: codex-npu-context
description: Use when a task is vague, underspecified, or would benefit from semantic retrieval over the user's local Codex context, prior Codex chats, indexed repositories, old decisions, errors, setup notes, or when exact filenames/keywords are unknown.
---

# Codex NPU Context

Use `codex_npu_search` when local semantic context is more useful than exact keyword search.

Good triggers:

- The user asks to retrieve prior decisions, old Codex chats, setup details, or "what did we do before?"
- The user describes something vaguely, from memory, or without the exact file, symbol, command, error text, or keyword.
- The relevant file name or exact keyword is uncertain.
- A task spans multiple repos, local tooling, setup notes, debugging history, or Codex config.
- `rg` is likely to miss synonyms or conversational context.

Workflow:

1. Call `codex_npu_status` if you need to confirm the index exists.
2. For vague local-memory lookups, run `codex_npu_search` with a concise natural-language query and, when concrete tokens exist, run `rg` exact search for those tokens in parallel.
3. Prefer `rg` for exact strings, symbols, filenames, commands, error text, and potential secrets.
4. Prefer MCP results for semantic leads when exact terms are missing, ambiguous, or too generic.
5. Treat returned paths and excerpts as leads, then read the real files before editing or making claims.
6. If MCP returns `no_confident_result` and exact search has no hits, say there is no local confident match and ask to re-index or search another source.

Important:

- This is a retrieval aid, not a source of truth.
- Use `rg` for exact strings, symbols, and filenames.
- Never print secret values. For credentials, tokens, passwords, or keys, report only the path and safe context unless the user explicitly asks to inspect the secret source.
- Rebuild the index after important new sessions, repo changes, or notes.
