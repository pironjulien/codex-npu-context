---
name: codex-npu-context
description: Use when a task is vague, underspecified, or would benefit from semantic retrieval over local Codex context, prior chats, indexed repositories, decisions, errors, setup notes, or when exact filenames and keywords are unknown. Requires the codex-npu-context MCP server.
---

# Codex NPU Context

Use the `codex_npu_search` MCP tool when local semantic context is more useful than exact keyword search.

Good triggers:

- The user asks to retrieve prior decisions, old Codex chats, setup details, or "what did we do before?"
- The user describes something vaguely, from memory, or without the exact file, symbol, command, error text, or keyword.
- The relevant file name or exact keyword is uncertain.
- A task spans multiple repos, local tooling, setup notes, debugging history, or Codex config.
- `rg` is likely to miss synonyms or conversational context.

Workflow:

1. Call `codex_npu_status` if you need to confirm the index exists.
2. For vague local-memory lookups with concrete tokens, prefer `codex_npu_dual_search`: use a concise natural-language `query`, pass caller-relevant `roots`, and put exact strings/symbols/regex tokens in `rg`.
3. If roots are unknown or no exact token exists, run `codex_npu_search` with a concise natural-language query and, when concrete tokens later appear, run `rg` exact search for those tokens in parallel.
4. Prefer `rg` for exact strings, symbols, filenames, commands, error text, and potential secrets.
5. Prefer MCP results for semantic leads when exact terms are missing, ambiguous, or too generic.
6. Treat returned paths and excerpts as leads, then read the real files before editing or making claims.
7. If MCP returns `no_confident_result` and exact search has no hits, say there is no local confident match and ask to re-index or search another source.

Important:

- This is a retrieval aid, not a source of truth.
- Use `rg` for exact strings, symbols, and filenames.
- Never print secret values. For credentials, tokens, passwords, or keys, report only the path and safe context unless the user explicitly asks to inspect the secret source.
- Rebuild the index after important new sessions, repo changes, or notes.
