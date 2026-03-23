# Cortex — AI Memory for Claude

**Persistent memory with a living knowledge graph. No API key. No cloud. 20/20 benchmark.**

Cortex is an open-source MCP server that gives Claude Desktop and Claude Code persistent memory across every conversation. It stores facts, detects contradictions, builds a typed knowledge graph, and runs a background scheduler that continuously finds patterns in your memories — all locally, all free.

```
npm run bench → 20/20 checks passed, 27/27 edge cases passed
```

---

## Why Cortex

| | Cortex | mem0 | basic-memory | kiro-memory |
|---|---|---|---|---|
| API key required | **None** | OpenAI (required) | None | None |
| Write latency | **3.7ms** | ~1–3s (LLM call) | ~5ms | ~5ms |
| Contradiction detection | **✓** | ✗ | ✗ | ✗ |
| Typed knowledge graph | **✓** | ✗ (flat vectors) | Markdown links | ✗ |
| Background scheduler | **✓** | ✗ | ✗ | ✗ |
| Token efficiency | **80% savings** | baseline | baseline | baseline |
| Local embeddings | **✓ MiniLM-L6** | OpenAI API | FastEmbed | ✗ |
| Claude Desktop | **✓** | ✓ | ✓ | ✓ |
| Claude Code | **✓** | ✗ | ✗ | ✗ |

---

## Install

Requires **Node.js v22.5+** (v24 recommended) — needed for `node:sqlite`.

### One-command setup (recommended)

```bash
npx @techiesgult/cortex-mem setup
```

This auto-configures Claude Desktop's MCP config and downloads the embedding model on first run. Restart Claude Desktop — Cortex is live.

### Or install globally

```bash
npm install -g @techiesgult/cortex-mem
cortex-mem setup
```

### Or install from source

```bash
git clone https://github.com/Aravindargutus/cortex-mem.git
cd cortex-mem
npm install && npm run build
node dist/cli.js setup
```

### Claude Code

```bash
claude mcp add cortex -- npx -y @techiesgult/cortex-mem serve
```

Or add manually to `~/.claude.json`:

```json
{
  "mcpServers": {
    "cortex": {
      "command": "npx",
      "args": ["-y", "@techiesgult/cortex-mem", "serve"]
    }
  }
}
```

### Verify it's working

In a Claude Desktop or Claude Code conversation:

```
Save this: I prefer TypeScript over JavaScript for all new projects.
```

Claude will call `memory_save` automatically. Next conversation:

```
What do you know about my language preferences?
```

Claude calls `memory_recall` and returns your saved preference.

---

## Benchmark

```
npm run bench
```

```
══════════════════════════════════════════════════════════════════════
  CORTEX BENCHMARK  —  2026-03-23  |  Node v24  |  darwin arm64
══════════════════════════════════════════════════════════════════════

  ── Embedding latency
  ✓ Embed (cold)            83ms    incl. model load
  ✓ Embed (warm avg)        2.8ms   n=20
  ✓ Embed (warm p99)        3.8ms

  ── Write performance
  ✓ Write throughput        271 mem/s

  ── Recall (vector only)
  ✓ Recall latency (avg)    3.2ms   n=5
  ✓ Recall Precision@5      72%

  ── Token efficiency
  ✓ Token savings           80.2%   vs full-context injection

  ── Contradiction detection
  ✓ Contradiction surface   100%    all 3 pairs found in rank ≤3

  ── Graph layer
  ✓ E2E recall (avg)        0.3ms   vector + 2-hop graph walk
  ✓ Graph walk coverage     60%     3/5 from 1 seed (2 hops)

  ── Correctness
  ✓ Supersede exclusion     100%    superseded memories fully hidden
  ✓ Timeline recall         60%     chronological order ✓
  ✓ Consolidation filters   100%    insight excl. ✓  superseded excl. ✓

  ── Scheduler / proactive
  ✓ Scheduler tick          78.7ms  28 memories → 1 cluster
  ✓ Cluster quality         100%    same-topic memories co-clustered
  ✓ Proactive CRUD          100%    save/list/review lifecycle ✓

  ── Edge cases              100%    27/27 passed

  SCORE: 20/20 checks passed
══════════════════════════════════════════════════════════════════════
```

---

## How It Works

### Memory save

When you tell Claude something worth remembering, Claude calls `memory_save`:

```
User: "I prefer dark mode in all my tools"
Claude: memory_save("User prefers dark mode", category="preference")
→ Embedded locally (MiniLM-L6-v2, 2.8ms)
→ Stored in SQLite with typed graph edges
→ Returns top 3 related memories so Claude can spot contradictions
```

### Memory recall

At the start of a new conversation (or when context would help):

```
Claude: memory_recall("tool preferences")
→ Vector search (3.2ms) + 2-hop graph walk (0.3ms)
→ Returns ~84 tokens of targeted memories (vs 427 for full injection)
→ 80% fewer tokens per query
```

### Contradiction detection

If you tell Claude something that contradicts an older memory:

```
Old memory: "User uses React for all frontend work"  [id:abc-123]
New:        "User switched to Vue.js after React fatigue"

→ memory_save returns [id:abc-123] as a related memory
→ Claude sees the conflict, calls memory_save(contradicts_id="abc-123")
→ Old memory marked superseded — never appears in future recalls
```

No API key used. Claude does this reasoning itself, using the returned IDs.

### Memory delete

When you ask Claude to forget something:

```
User: "Actually, forget that I prefer dark mode — I switched to light."
Claude: memory_delete(memory_id="abc-123", confirm=true)
→ Deletes memory + embedding + graph edges (atomically)
→ Scrubs from any pending clusters
→ If this memory contradicted an older one, restores the older one
→ "Deleted. Restored 1 previously-superseded memory."
```

Also available from the CLI:

```bash
cortex-mem forget abc-123-uuid
```

### Background scheduler

Cortex runs a background scheduler that clusters your memories every 30 minutes. When you start a new conversation, Claude can call `memory_proactive` to retrieve pre-computed pattern groups and synthesise them into insights.

```
memory_proactive()
→ "Found 2 patterns while you were away:
   Pattern 1 (4 memories): You consistently choose zero-infra tools
   Pattern 2 (3 memories): You prefer TypeScript in all new projects"
```

Zero active sessions required — also works standalone:

```bash
# Run one background pass
node dist/cli.js daemon --once

# Run continuously (e.g., register with launchd)
node dist/cli.js daemon
```

---

## MCP Tools

| Tool | When Claude calls it |
|---|---|
| `memory_save` | Anytime something worth remembering is mentioned |
| `memory_delete` | When user asks to forget something or remove incorrect info |
| `memory_recall` | Beginning of conversation, or when context helps |
| `memory_timeline` | "How did my thinking on X change over time?" |
| `memory_consolidate` | On-demand: cluster and synthesise recent memories |
| `memory_proactive` | Pull pre-computed patterns from background scheduler |

---

## Configuration

Config lives at `~/.cortex/config.json`:

```json
{
  "dbPath": "~/.cortex/memory.db",
  "embeddingMode": "local",
  "schedulerIntervalMs": 1800000
}
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "cortex": {
      "command": "npx",
      "args": ["-y", "@techiesgult/cortex-mem", "serve"]
    }
  }
}
```

**Claude Code** (`~/.claude.json`):

```json
{
  "mcpServers": {
    "cortex": {
      "command": "npx",
      "args": ["-y", "@techiesgult/cortex-mem", "serve"]
    }
  }
}
```

> **Note:** Claude Desktop uses its own Node binary which may be v20. Ensure Node v22.5+ is available on your PATH. Find yours: `which node` after `nvm use 24`.

---

## CLI

```bash
cortex-mem setup                  # Auto-configure Claude Desktop + Claude Code
cortex-mem serve                  # Start MCP server (Claude does this automatically)
cortex-mem stats                  # Show memory count + pending patterns
cortex-mem forget <uuid>          # Delete a memory by ID
cortex-mem daemon                 # Run background scheduler continuously
cortex-mem daemon --once          # Single scheduler tick and exit
cortex-mem daemon --interval=300  # Tick every 5 minutes
```

All commands also work with `npx @techiesgult/cortex-mem <command>`.

---

## Stats

```bash
node dist/cli.js stats
```

```
  🧠 Cortex Memory Stats
  Memories:         142
  Pending patterns: 3
  Database:         ~/.cortex/memory.db
  Embedding:        local
```

---

## Architecture

```
Claude Desktop (MCP client)
       │
       ▼
CortexMcpServer (stdio MCP)
  ├── memory_save        → CortexStorage.saveMemory()
  │                         └── sqlite-vec (vec0 virtual table, 384-dim)
  ├── memory_delete      → CortexStorage.deleteMemory()
  │                         └── cascade edges + vec0 + clusters + supersede restore
  ├── memory_recall      → vectorSearch() + graphWalk()
  ├── memory_timeline    → getTimelineMemories()
  ├── memory_consolidate → embed+cluster on-demand
  └── memory_proactive   → getUnreviewedClusters()

CortexScheduler (background, every 30 min)
  └── tick() → embed + cluster → savePendingCluster()

Storage: ~/.cortex/memory.db (SQLite)
  ├── memories           (content, category, source, superseded)
  ├── edges              (typed: relates_to / contradicts / evolved_from / ...)
  ├── memory_vectors     (vec0 virtual table, 384-dim float32)
  ├── pending_clusters   (background scheduler output)
  └── consolidation_meta (last run timestamp)

Embeddings: @xenova/transformers — Xenova/all-MiniLM-L6-v2 (ONNX, quantized)
            ~/.cortex/models/ — downloaded once, no network after that
```

---

## Requirements

- **Node.js v22.5+** (v24 recommended) — required for `node:sqlite`
- macOS or Linux (Windows untested)
- Claude Desktop or Claude Code

---

## License

MIT
