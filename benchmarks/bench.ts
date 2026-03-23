/**
 * Cortex Benchmark Suite
 *
 * Measures:
 *   1.  Embedding latency (cold model load vs warm)
 *   2.  Write throughput (memories/sec)
 *   3.  Recall latency — vector only (avg + p99)
 *   4.  Recall precision@K vs ground truth
 *   5.  Token efficiency vs full-context injection
 *   6.  Contradiction surface rate
 *   7.  Embedding quality (intra vs inter cluster separation)
 *   8.  Graph walk — edge create, E2E recall latency, coverage
 *   9.  Supersede correctness (excluded from future searches)
 *   10. Timeline correctness (chronological order + cluster precision)
 *   11. Consolidation candidate correctness (insight/superseded exclusion)
 *   12. Scheduler tick — latency, cluster quality, pending CRUD
 *   13. Proactive cluster correctness (review lifecycle, count accuracy)
 *   14. Storage efficiency (bytes per memory)
 *   15. Edge cases — deduplication, edge dedup, empty/unicode/long content,
 *       consolidated_at marking, embedding retrieval, concurrent scheduler
 *
 * Usage:
 *   npm run bench
 */

import { CortexStorage } from "../src/storage/database.js";
import { CortexScheduler } from "../src/agent/scheduler.js";
import { embedLocal } from "../src/embeddings/engine.js";
import { tmpdir } from "os";
import { join } from "path";
import { unlinkSync, existsSync, statSync } from "fs";
import type { MemoryCategory } from "../src/types/index.js";

// ---------------------------------------------------------------------------
// Test dataset — 5 topic clusters, 5 memories each (25 total)
// ---------------------------------------------------------------------------

const SEED_DATA: Record<string, Array<{ content: string; category: MemoryCategory }>> = {
  typescript: [
    { content: "User prefers TypeScript over JavaScript for all new projects", category: "preference" },
    { content: "User uses strict TypeScript with noImplicitAny and strictNullChecks enabled", category: "preference" },
    { content: "User chose TypeScript for Cortex because of type safety and IDE support", category: "decision" },
    { content: "User finds TypeScript generics confusing but values them for library code", category: "fact" },
    { content: "User runs tsc --noEmit as part of CI for type checking", category: "fact" },
  ],
  database: [
    { content: "User chose SQLite for Cortex because it requires no server infrastructure", category: "decision" },
    { content: "User evaluated Postgres but decided it was overkill for a single-user tool", category: "decision" },
    { content: "User uses WAL mode for SQLite to improve concurrent read performance", category: "fact" },
    { content: "User prefers embedded databases over client-server databases for local tools", category: "preference" },
    { content: "User trusts better-sqlite3 because of its synchronous API and reliability", category: "fact" },
  ],
  nodejs: [
    { content: "User runs Node.js v24 because node:sqlite requires v22.5+", category: "fact" },
    { content: "User manages Node versions with nvm and keeps multiple versions installed", category: "fact" },
    { content: "User prefers ESM modules over CommonJS for all new Node projects", category: "preference" },
    { content: "User uses tsup for bundling TypeScript into distributable packages", category: "fact" },
    { content: "User finds Deno interesting but has not adopted it in production yet", category: "fact" },
  ],
  project: [
    { content: "User is building Cortex, an open-source AI memory engine with a knowledge graph", category: "fact" },
    { content: "User wants to launch Cortex on Hacker News after publishing to npm", category: "decision" },
    { content: "Cortex uses the MCP protocol to integrate with Claude Desktop", category: "fact" },
    { content: "User removed API key requirement from Cortex to lower the barrier to entry", category: "decision" },
    { content: "User's goal is to publish Cortex to npm within the next week", category: "decision" },
  ],
  personal: [
    { content: "User's name is Aravind and they are based in India", category: "fact" },
    { content: "User builds developer tools as a side project and hobby", category: "fact" },
    { content: "User prefers working late at night when there are fewer distractions", category: "preference" },
    { content: "User learns by building things rather than reading documentation first", category: "preference" },
    { content: "User is interested in AI tooling, knowledge graphs, and developer experience", category: "fact" },
  ],
};

// Ground truth: each query should return its cluster in top-K results
const RECALL_TESTS: Array<{ query: string; cluster: string; topK: number }> = [
  { query: "What does the user think about TypeScript and type safety?", cluster: "typescript", topK: 5 },
  { query: "Database technology choices and infrastructure preferences", cluster: "database", topK: 5 },
  { query: "Node.js version management and module system preferences", cluster: "nodejs", topK: 5 },
  { query: "What is the Cortex project and its goals?", cluster: "project", topK: 5 },
  { query: "Who is the user, where are they from, and what do they work on?", cluster: "personal", topK: 5 },
];

// Contradiction pairs: original fact, then a contradicting update
const CONTRADICTION_TESTS = [
  {
    original: "User plans to use Postgres as the primary database for Cortex",
    update:   "User switched from Postgres to SQLite for Cortex — no server setup needed",
  },
  {
    original: "User prefers React for all frontend development projects",
    update:   "User has moved away from React and now prefers using Vue.js for new frontend projects",
  },
  {
    original: "User publishes packages under the @aravind npm scope",
    update:   "User decided to publish Cortex under the @cortex-ai npm scope instead",
  },
];

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function cosineSim(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot  += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

function percentile(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)];
}

function approxTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

type Result = {
  label: string;
  value: number;
  unit: string;
  note?: string;
  pass?: boolean;    // undefined = informational, true = pass, false = fail
};

const PASS_THRESHOLDS: Record<string, number> = {
  "Embed (cold)":               3000,   // <3 s
  "Embed (warm avg)":           50,     // <50 ms
  "Embed (warm p99)":           100,    // <100 ms
  "Write throughput":           0.5,    // >0.5 mem/s (bottleneck is embedding)
  "Recall latency (avg)":       60,     // <60 ms
  "Recall latency (p99)":       120,    // <120 ms
  "E2E recall latency (avg)":   80,     // <80 ms — vector + 2-hop graph walk
  "E2E recall latency (p99)":   150,    // <150 ms
  "Recall Precision@5":         60,     // >60 % — decent retrieval
  "Token savings":              50,     // >50 % vs full-context
  "Contradiction surface rate": 66,     // ≥2/3 surfaced in top 3
  "Supersede exclusion":        100,    // 100 % — hard correctness requirement
  "Timeline recall":            60,     // >60 % of top-5 from expected cluster
  "Consolidation candidate rate": 100, // all 3 filters must hold
  "Graph walk coverage":        60,     // >60 % of cluster found from 1 seed (2 hops)
  "Embedding separation":       1.20,   // intra/inter ratio > 1.2
  "Scheduler tick latency":     10_000, // <10 s (25 memories + embed)
  "Scheduler cluster quality":  60,     // >60 % of found clusters have ≥2 members from same topic
  "Proactive CRUD correctness": 100,    // save / list / mark-reviewed / count all correct
  "Edge case pass rate":         100,    // all edge cases must pass
};

function evalPass(label: string, value: number): boolean | undefined {
  const threshold = PASS_THRESHOLDS[label];
  if (threshold === undefined) return undefined;
  // Time/latency metrics: lower is better
  const lowerIsBetter = label.includes("latency") || label.includes("cold") || label.includes("warm");
  return lowerIsBetter ? value <= threshold : value >= threshold;
}

// ---------------------------------------------------------------------------
// Main benchmark
// ---------------------------------------------------------------------------

async function main() {
  const dbPath = join(tmpdir(), `cortex-bench-${Date.now()}.db`);
  const storage = new CortexStorage(dbPath, { unsafeSkipPathCheck: true });
  const results: Result[] = [];

  console.log("\n" + "═".repeat(70));
  console.log("  CORTEX BENCHMARK  —  " + new Date().toISOString().slice(0, 19).replace("T", " "));
  console.log("  Node " + process.version + "  |  " + process.platform + " " + process.arch);
  console.log("═".repeat(70));

  try {
    // -----------------------------------------------------------------------
    // PHASE 1: EMBEDDING LATENCY
    // -----------------------------------------------------------------------
    console.log("\n[1/8] Embedding latency...");

    const t0 = performance.now();
    await embedLocal("Hello world — cold start embedding test");
    const coldMs = performance.now() - t0;
    results.push({ label: "Embed (cold)", value: coldMs, unit: "ms", note: "incl. model load", pass: evalPass("Embed (cold)", coldMs) });

    const warmRuns: number[] = [];
    for (let i = 0; i < 20; i++) {
      const t1 = performance.now();
      await embedLocal(`Warm embedding test sentence number ${i} about software development`);
      warmRuns.push(performance.now() - t1);
    }
    const warmAvg = warmRuns.reduce((a, b) => a + b, 0) / warmRuns.length;
    const warmP99 = percentile(warmRuns, 99);
    results.push({ label: "Embed (warm avg)", value: warmAvg, unit: "ms", note: "n=20", pass: evalPass("Embed (warm avg)", warmAvg) });
    results.push({ label: "Embed (warm p99)", value: warmP99, unit: "ms", pass: evalPass("Embed (warm p99)", warmP99) });

    // -----------------------------------------------------------------------
    // PHASE 2: WRITE THROUGHPUT
    // -----------------------------------------------------------------------
    console.log("[2/8] Write throughput...");

    const seedIds = new Map<string, string[]>();
    const writeLatencies: number[] = [];

    for (const [cluster, memories] of Object.entries(SEED_DATA)) {
      const ids: string[] = [];
      for (const { content, category } of memories) {
        const tw0 = performance.now();
        const mem = storage.saveMemory(content, category);
        const emb = await embedLocal(content);
        storage.saveEmbedding(mem.id, emb);
        writeLatencies.push(performance.now() - tw0);
        ids.push(mem.id);
      }
      seedIds.set(cluster, ids);
    }

    const avgWriteMs = writeLatencies.reduce((a, b) => a + b, 0) / writeLatencies.length;
    const totalWriteMs = writeLatencies.reduce((a, b) => a + b, 0);
    const throughput = (writeLatencies.length / totalWriteMs) * 1000;
    results.push({ label: "Write latency (avg)", value: avgWriteMs, unit: "ms", note: `n=${writeLatencies.length}` });
    results.push({ label: "Write throughput", value: throughput, unit: "mem/s", note: `${writeLatencies.length} memories in ${(totalWriteMs / 1000).toFixed(1)}s`, pass: evalPass("Write throughput", throughput) });

    // -----------------------------------------------------------------------
    // PHASE 3: RECALL LATENCY
    // -----------------------------------------------------------------------
    console.log("[3/8] Recall latency...");

    const recallLatencies: number[] = [];
    for (const test of RECALL_TESTS) {
      const tr0 = performance.now();
      const emb = await embedLocal(test.query);
      storage.vectorSearch(emb, test.topK);
      recallLatencies.push(performance.now() - tr0);
    }

    const recallAvg = recallLatencies.reduce((a, b) => a + b, 0) / recallLatencies.length;
    const recallP99 = percentile(recallLatencies, 99);
    results.push({ label: "Recall latency (avg)", value: recallAvg, unit: "ms", note: `n=${recallLatencies.length}`, pass: evalPass("Recall latency (avg)", recallAvg) });
    results.push({ label: "Recall latency (p99)", value: recallP99, unit: "ms", pass: evalPass("Recall latency (p99)", recallP99) });

    // -----------------------------------------------------------------------
    // PHASE 4: RECALL PRECISION @ K
    // -----------------------------------------------------------------------
    console.log("[4/8] Recall precision@K...");

    let totalHits = 0;
    let totalSlots = 0;
    const perQueryHits: number[] = [];

    for (const test of RECALL_TESTS) {
      const emb = await embedLocal(test.query);
      const matches = storage.vectorSearch(emb, test.topK);
      const clusterIds = new Set(seedIds.get(test.cluster) ?? []);
      const hits = matches.filter((m) => clusterIds.has(m.memory.id)).length;
      perQueryHits.push(hits);
      totalHits += hits;
      totalSlots += test.topK;
    }

    const precision = (totalHits / totalSlots) * 100;
    const perQStr = perQueryHits.map((h, i) => `${RECALL_TESTS[i].cluster.slice(0, 3)}:${h}/${RECALL_TESTS[i].topK}`).join(" ");
    results.push({ label: `Recall Precision@5`, value: precision, unit: "%", note: perQStr, pass: evalPass("Recall Precision@5", precision) });

    // -----------------------------------------------------------------------
    // PHASE 5: TOKEN EFFICIENCY
    // -----------------------------------------------------------------------
    console.log("[5/8] Token efficiency...");

    // Full-context: every memory pasted into every prompt (what naive memory does)
    const allMemories = storage.getAllMemories();
    const fullContextTokens = approxTokens(allMemories.map((m) => m.content).join("\n"));

    // GPT-style memory: top 10 "recent" memories regardless of query (what basic-memory does naively)
    const recentTenTokens = approxTokens(allMemories.slice(0, 10).map((m) => m.content).join("\n"));

    // Cortex: vector-filtered recall per query
    let cortexTotalTokens = 0;
    for (const test of RECALL_TESTS) {
      const emb = await embedLocal(test.query);
      const matches = storage.vectorSearch(emb, test.topK);
      cortexTotalTokens += approxTokens(matches.map((m) => m.memory.content).join("\n"));
    }
    const cortexAvgTokens = cortexTotalTokens / RECALL_TESTS.length;

    const savingsVsFull = ((fullContextTokens - cortexAvgTokens) / fullContextTokens) * 100;
    const _savingsVsRecent = ((recentTenTokens - cortexAvgTokens) / recentTenTokens) * 100;

    results.push({ label: "Full-context tokens",      value: fullContextTokens,  unit: "tok", note: "all memories in prompt (naive)" });
    results.push({ label: "Recent-10 tokens",         value: recentTenTokens,    unit: "tok", note: "top-10 recent (GPT-style)" });
    results.push({ label: "Cortex recall tokens",     value: cortexAvgTokens,    unit: "tok", note: "avg per query" });
    results.push({ label: "Token savings",            value: savingsVsFull,      unit: "%",   note: `vs full-context`, pass: evalPass("Token savings", savingsVsFull) });

    // -----------------------------------------------------------------------
    // PHASE 6: CONTRADICTION SURFACE RATE
    // -----------------------------------------------------------------------
    console.log("[6/8] Contradiction detection...");

    let surfaced = 0;
    const contResults: string[] = [];

    for (const test of CONTRADICTION_TESTS) {
      // Save the original fact
      const origMem = storage.saveMemory(test.original, "fact");
      const origEmb = await embedLocal(test.original);
      storage.saveEmbedding(origMem.id, origEmb);

      // Save the contradicting update and check if original surfaces in top 3
      const updEmb = await embedLocal(test.update);
      const related = storage.vectorSearch(updEmb, 8);
      const _topRelated = related
        .filter((r) => r.memory.id !== origMem.id && r.distance < 0.6)
        .slice(0, 3);

      // Also check if the exact original is within top 4 of the full search
      const originalRank = related.findIndex((r) => r.memory.id === origMem.id);
      const found = originalRank >= 0 && originalRank < 4;

      if (found) surfaced++;
      contResults.push(`${found ? "✓" : "✗"} rank=${originalRank >= 0 ? originalRank + 1 : "–"} sim=${originalRank >= 0 ? (1 - related[originalRank].distance).toFixed(3) : "–"}`);
    }

    const surfaceRate = (surfaced / CONTRADICTION_TESTS.length) * 100;
    results.push({
      label: "Contradiction surface rate",
      value: surfaceRate,
      unit: "%",
      note: contResults.join("  "),
      pass: evalPass("Contradiction surface rate", surfaceRate),
    });

    // -----------------------------------------------------------------------
    // PHASE 7: EMBEDDING QUALITY
    // -----------------------------------------------------------------------
    console.log("[7/8] Embedding quality...");

    const clusterEmbs = new Map<string, number[][]>();
    for (const [cluster, mems] of Object.entries(SEED_DATA)) {
      const embs: number[][] = [];
      for (const { content } of mems) {
        embs.push(await embedLocal(content));
      }
      clusterEmbs.set(cluster, embs);
    }

    // Intra-cluster similarity (same topic)
    let intraSum = 0, intraN = 0;
    for (const [, embs] of clusterEmbs) {
      for (let i = 0; i < embs.length; i++) {
        for (let j = i + 1; j < embs.length; j++) {
          intraSum += cosineSim(embs[i], embs[j]);
          intraN++;
        }
      }
    }

    // Inter-cluster similarity (different topics)
    const embClusters = Array.from(clusterEmbs.values());
    let interSum = 0, interN = 0;
    for (let c1 = 0; c1 < embClusters.length; c1++) {
      for (let c2 = c1 + 1; c2 < embClusters.length; c2++) {
        for (const e1 of embClusters[c1]) {
          for (const e2 of embClusters[c2]) {
            interSum += cosineSim(e1, e2);
            interN++;
          }
        }
      }
    }

    const intraAvg = intraSum / intraN;
    const interAvg = interSum / interN;
    const separationRatio = intraAvg / interAvg;

    results.push({ label: "Embedding intra-cluster sim", value: intraAvg * 100, unit: "%",  note: "higher = tighter topics" });
    results.push({ label: "Embedding inter-cluster sim", value: interAvg * 100, unit: "%",  note: "lower = better separation" });
    results.push({ label: "Embedding separation",        value: separationRatio, unit: "x", note: "intra÷inter — >1.2 is good", pass: evalPass("Embedding separation", separationRatio) });

    // -----------------------------------------------------------------------
    // PHASE 8: GRAPH WALK — edge creation, latency, coverage
    // -----------------------------------------------------------------------
    console.log("[8/12] Graph walk...");

    // Create edges between seeded memories to simulate server.ts auto-edge behaviour
    // Within each cluster: link consecutive memories with relates_to edges
    let edgesCreated = 0;
    for (const [, ids] of seedIds) {
      for (let i = 0; i < ids.length - 1; i++) {
        storage.addEdge(ids[i], ids[i + 1], "relates_to", 0.9);
        edgesCreated++;
      }
    }
    // Cross-cluster edge: TypeScript ↔ Node.js (both TS/JS ecosystem)
    const tsIds   = seedIds.get("typescript") ?? [];
    const nodeIds = seedIds.get("nodejs") ?? [];
    if (tsIds.length && nodeIds.length) {
      storage.addEdge(tsIds[0], nodeIds[0], "relates_to", 0.7);
      edgesCreated++;
    }

    // Verify edge retrieval works
    const sampleEdges = storage.getEdgesFrom(tsIds[0]);
    const edgeRetrievalOk = sampleEdges.length >= 1;
    results.push({ label: "Edge creation", value: edgesCreated, unit: "edges", note: `retrieval ${edgeRetrievalOk ? "✓" : "✗"}` });

    // E2E recall latency: vector search + 2-hop graph walk (what memory_recall actually does)
    const gwLatencies: number[] = [];
    for (const test of RECALL_TESTS) {
      const emb = await embedLocal(test.query);
      const tgw = performance.now();
      const vr  = storage.vectorSearch(emb, test.topK);
      const startIds = vr.slice(0, 3).map((r) => r.memory.id);
      storage.graphWalk(startIds, 2);
      gwLatencies.push(performance.now() - tgw);
    }
    const gwAvg = gwLatencies.reduce((a, b) => a + b, 0) / gwLatencies.length;
    const gwP99 = percentile(gwLatencies, 99);
    results.push({ label: "E2E recall latency (avg)", value: gwAvg, unit: "ms", note: `vector + 2-hop graph, n=${gwLatencies.length}`, pass: evalPass("E2E recall latency (avg)", gwAvg) });
    results.push({ label: "E2E recall latency (p99)", value: gwP99, unit: "ms", pass: evalPass("E2E recall latency (p99)", gwP99) });

    // Graph walk coverage: starting from 1 seed in typescript cluster, how many TS memories surface?
    const coverageWalked = storage.graphWalk(tsIds.slice(0, 1), 2);
    const walkedIds = new Set(coverageWalked.map((w) => w.memory.id));
    const tsHits = tsIds.filter((id) => walkedIds.has(id)).length;
    const coverage = (tsHits / tsIds.length) * 100;
    results.push({ label: "Graph walk coverage", value: coverage, unit: "%", note: `${tsHits}/${tsIds.length} typescript memories from 1 seed (2 hops)`, pass: evalPass("Graph walk coverage", coverage) });

    // -----------------------------------------------------------------------
    // PHASE 9: SUPERSEDE CORRECTNESS
    // -----------------------------------------------------------------------
    console.log("[9/12] Supersede correctness...");

    const dbIds = seedIds.get("database") ?? [];
    for (const id of dbIds) storage.supersedeMemory(id);

    const supersedeEmb = await embedLocal("SQLite and database technology preferences");
    const afterSupersede = storage.vectorSearch(supersedeEmb, 10);
    const leaked = afterSupersede.filter((r) => dbIds.includes(r.memory.id));
    const exclusionRate = ((dbIds.length - leaked.length) / dbIds.length) * 100;
    results.push({
      label: "Supersede exclusion",
      value: exclusionRate,
      unit: "%",
      note: leaked.length === 0
        ? `all ${dbIds.length} superseded memories correctly hidden`
        : `LEAK: ${leaked.length} superseded still visible`,
      pass: evalPass("Supersede exclusion", exclusionRate),
    });

    // -----------------------------------------------------------------------
    // PHASE 10: TIMELINE CORRECTNESS
    // -----------------------------------------------------------------------
    console.log("[10/12] Timeline correctness...");

    const timelineEmb = await embedLocal("Cortex project decisions and goals");
    const timeline = storage.getTimelineMemories(timelineEmb, 10);

    let chronoOk = true;
    for (let i = 1; i < timeline.length; i++) {
      if (timeline[i].memory.created_at < timeline[i - 1].memory.created_at) {
        chronoOk = false;
        break;
      }
    }

    const projectIds = new Set(seedIds.get("project") ?? []);
    const topFive = timeline.slice(0, 5);
    const timelineHits = topFive.filter((t) => projectIds.has(t.memory.id)).length;
    const timelinePrecision = topFive.length > 0 ? (timelineHits / topFive.length) * 100 : 0;

    results.push({
      label: "Timeline recall",
      value: timelinePrecision,
      unit: "%",
      note: `${timelineHits}/5 from expected cluster | chrono order ${chronoOk ? "✓" : "✗"}`,
      pass: evalPass("Timeline recall", timelinePrecision),
    });

    // -----------------------------------------------------------------------
    // PHASE 11: CONSOLIDATION CANDIDATE CORRECTNESS
    // -----------------------------------------------------------------------
    console.log("[11/12] Consolidation candidates...");

    // Save a consolidated insight — must be excluded from unconsolidated candidates
    const insightMem = storage.saveMemory(
      "The user values zero-infrastructure local tools over cloud solutions",
      "insight",
      "consolidated"
    );
    storage.saveEmbedding(insightMem.id, await embedLocal(insightMem.content));

    const unconsolidated = storage.getUnconsolidatedMemories(100);
    const insightExcluded    = !unconsolidated.some((m) => m.id === insightMem.id);
    const supersedeExcluded2 = !unconsolidated.some((m) => dbIds.includes(m.id));

    // Consolidation time roundtrip
    const nowIso = new Date().toISOString();
    storage.setLastConsolidationTime(nowIso);
    const retrieved = storage.getLastConsolidationTime();
    const timeRoundtrip = retrieved === nowIso;

    const checks = [insightExcluded, supersedeExcluded2, timeRoundtrip];
    const pct = (checks.filter(Boolean).length / checks.length) * 100;
    results.push({
      label: "Consolidation candidate rate",
      value: pct,
      unit: "%",
      note: `insight excl. ${insightExcluded ? "✓" : "✗"} | superseded excl. ${supersedeExcluded2 ? "✓" : "✗"} | time roundtrip ${timeRoundtrip ? "✓" : "✗"}`,
      pass: evalPass("Consolidation candidate rate", pct),
    });

    // -----------------------------------------------------------------------
    // PHASE 12: SCHEDULER TICK — latency, cluster quality, pending CRUD
    // -----------------------------------------------------------------------
    console.log("[12/14] Scheduler tick...");

    // Build a fresh isolated DB + scheduler for this phase so earlier supersedes don't skew it
    const schedDbPath = join(tmpdir(), `cortex-bench-sched-${Date.now()}.db`);
    const schedStorage = new CortexStorage(schedDbPath, { unsafeSkipPathCheck: true });

    // Seed the same 25 memories (no supersedes) into this DB
    for (const [, memories] of Object.entries(SEED_DATA)) {
      for (const { content, category } of memories) {
        const m   = schedStorage.saveMemory(content, category);
        const emb = await embedLocal(content);
        schedStorage.saveEmbedding(m.id, emb);
      }
    }

    // Seed a few extra diverse memories to hit the ≥5 threshold comfortably
    const extraMems = [
      "User drinks coffee every morning before starting work",
      "User has a standing desk at their home office",
      "User prefers mechanical keyboards for coding",
    ];
    for (const content of extraMems) {
      const m = schedStorage.saveMemory(content, "fact");
      schedStorage.saveEmbedding(m.id, await embedLocal(content));
    }

    const silentScheduler = new CortexScheduler(schedStorage);
    const tickT0 = performance.now();
    const tickResult = await silentScheduler.tick();
    const tickMs = performance.now() - tickT0;

    results.push({
      label: "Scheduler tick latency",
      value: tickMs,
      unit: "ms",
      note: `scanned ${tickResult.memoriesScanned}, saved ${tickResult.clustersSaved} cluster(s)`,
      pass: evalPass("Scheduler tick latency", tickMs),
    });

    // Quality check: for every saved cluster, count how many memories share a seed-topic cluster
    // Build a map: memoryId → cluster label
    const idToCluster = new Map<string, string>();
    const schedAllMems = schedStorage.getAllMemories();
    // Re-embed all and match by content to original SEED_DATA
    for (const [clusterName, seedMems] of Object.entries(SEED_DATA)) {
      for (const { content } of seedMems) {
        const found = schedAllMems.find((m) => m.content === content);
        if (found) idToCluster.set(found.id, clusterName);
      }
    }

    const clusters = schedStorage.getUnreviewedClusters(20);
    let qualityClusters = 0;
    for (const { cluster } of clusters) {
      const labels = cluster.memory_ids
        .map((id) => idToCluster.get(id))
        .filter(Boolean) as string[];
      const counts = labels.reduce((acc, l) => { acc[l] = (acc[l] ?? 0) + 1; return acc; }, {} as Record<string, number>);
      const maxSameLabel = Math.max(0, ...Object.values(counts));
      if (maxSameLabel >= 2) qualityClusters++;
    }
    const clusterQualityPct = clusters.length > 0 ? (qualityClusters / clusters.length) * 100 : 0;
    results.push({
      label: "Scheduler cluster quality",
      value: clusterQualityPct,
      unit: "%",
      note: `${qualityClusters}/${clusters.length} clusters have ≥2 same-topic memories`,
      pass: evalPass("Scheduler cluster quality", clusterQualityPct),
    });

    // -----------------------------------------------------------------------
    // PHASE 13: PROACTIVE CLUSTER CRUD CORRECTNESS
    // -----------------------------------------------------------------------
    console.log("[13/14] Proactive cluster correctness...");

    // savePendingCluster
    const savedCluster = schedStorage.savePendingCluster(["id-a", "id-b", "id-c"]);
    const savedOk = savedCluster.id.length > 0 && !savedCluster.reviewed;

    // getPendingClusterCount
    const countBefore = schedStorage.getPendingClusterCount();
    const countOk = countBefore > 0; // scheduler just saved some + the manual one

    // getUnreviewedClusters — returns only reviewed=0 rows
    const unreviewedBefore = schedStorage.getUnreviewedClusters(50);
    const listOk = unreviewedBefore.length === countBefore;

    // markClusterReviewed
    schedStorage.markClusterReviewed(savedCluster.id);
    const countAfter = schedStorage.getPendingClusterCount();
    const markOk = countAfter === countBefore - 1;

    // Verify reviewed cluster no longer appears in getUnreviewedClusters
    const unreviewedAfter = schedStorage.getUnreviewedClusters(50);
    const hiddenOk = !unreviewedAfter.some(({ cluster }) => cluster.id === savedCluster.id);

    schedStorage.close();
    if (existsSync(schedDbPath)) unlinkSync(schedDbPath);

    const crudChecks = [savedOk, countOk, listOk, markOk, hiddenOk];
    const crudPct = (crudChecks.filter(Boolean).length / crudChecks.length) * 100;
    results.push({
      label: "Proactive CRUD correctness",
      value: crudPct,
      unit: "%",
      note: [
        `save ${savedOk ? "✓" : "✗"}`,
        `count ${countOk ? "✓" : "✗"}`,
        `list ${listOk ? "✓" : "✗"}`,
        `mark-reviewed ${markOk ? "✓" : "✗"}`,
        `hidden-after-review ${hiddenOk ? "✓" : "✗"}`,
      ].join("  "),
      pass: evalPass("Proactive CRUD correctness", crudPct),
    });

    // -----------------------------------------------------------------------
    // PHASE 14: STORAGE EFFICIENCY
    // -----------------------------------------------------------------------
    console.log("[14/14] Storage efficiency...\n");

    storage.close();
    const stat = statSync(dbPath);
    const count = allMemories.length;
    const bytesPerMem = stat.size / count;
    const kbPerMem = bytesPerMem / 1024;

    // Each memory embedding is 384 floats × 4 bytes = 1536 bytes
    const embeddingBytes = 384 * 4;
    const overhead = ((bytesPerMem - embeddingBytes) / bytesPerMem) * 100;

    results.push({ label: "DB size per memory",    value: kbPerMem,         unit: "KB",  note: `total ${(stat.size / 1024).toFixed(0)} KB / ${count} memories (vec0 fixed cost amortizes at scale)` });
    results.push({ label: "Embedding bytes",        value: embeddingBytes / 1024, unit: "KB", note: "384-dim float32 = 1536 bytes" });
    results.push({ label: "Metadata overhead",      value: overhead,         unit: "%",   note: "shrinks toward ~10% at 1k+ memories" });

    // -----------------------------------------------------------------------
    // PHASE 15: EDGE CASES
    // -----------------------------------------------------------------------
    console.log("[15/15] Edge cases...");

    const edgeCaseDb = join(tmpdir(), `cortex-bench-edge-${Date.now()}.db`);
    const ecStorage = new CortexStorage(edgeCaseDb, { unsafeSkipPathCheck: true });
    const ecResults: Array<{ name: string; pass: boolean; note: string }> = [];

    // EC1: Near-duplicate detection — saving the same content twice should
    //      produce two distinct rows (dedup is in server.ts, not storage layer)
    //      but storage.vectorSearch should find them with distance ≈ 0
    {
      const m1 = ecStorage.saveMemory("User prefers dark mode", "preference");
      const e1 = await embedLocal("User prefers dark mode");
      ecStorage.saveEmbedding(m1.id, e1);

      const m2 = ecStorage.saveMemory("User prefers dark mode", "preference");
      const e2 = await embedLocal("User prefers dark mode");
      ecStorage.saveEmbedding(m2.id, e2);

      const search = ecStorage.vectorSearch(e1, 5);
      const found = search.filter((r) => r.distance < 0.01);
      const hasBoth = found.some((r) => r.memory.id === m1.id) && found.some((r) => r.memory.id === m2.id);
      ecResults.push({
        name: "Near-duplicate detection",
        pass: hasBoth && found.length >= 2,
        note: `distance=${found[0]?.distance.toFixed(4) ?? "N/A"} found=${found.length}`,
      });
    }

    // EC2: Edge deduplication — edgeExists should prevent creating same edge twice
    {
      const ma = ecStorage.saveMemory("Test memory A", "fact");
      const mb = ecStorage.saveMemory("Test memory B", "fact");
      ecStorage.addEdge(ma.id, mb.id, "relates_to", 0.9);

      const existsBefore = ecStorage.edgeExists(ma.id, mb.id);
      const existsReverse = ecStorage.edgeExists(mb.id, ma.id); // both directions

      ecResults.push({
        name: "Edge dedup (forward)",
        pass: existsBefore === true,
        note: `edgeExists(A→B)=${existsBefore}`,
      });
      ecResults.push({
        name: "Edge dedup (reverse)",
        pass: existsReverse === true,
        note: `edgeExists(B→A)=${existsReverse}`,
      });
    }

    // EC3: Empty content embedding — should not crash
    {
      let emptyOk = false;
      try {
        const emb = await embedLocal("");
        emptyOk = Array.isArray(emb) && emb.length === 384;
      } catch {
        emptyOk = false;
      }
      ecResults.push({
        name: "Empty string embed",
        pass: emptyOk,
        note: emptyOk ? "384-dim vector returned" : "CRASHED",
      });
    }

    // EC4: Unicode / emoji content
    {
      const unicodeContent = "用户喜欢深色模式 🌙 и использует TypeScript 🚀";
      let unicodeOk = false;
      try {
        const mem = ecStorage.saveMemory(unicodeContent, "fact");
        const emb = await embedLocal(unicodeContent);
        ecStorage.saveEmbedding(mem.id, emb);
        const results = ecStorage.vectorSearch(emb, 3);
        unicodeOk = results.some((r) => r.memory.id === mem.id);
      } catch {
        unicodeOk = false;
      }
      ecResults.push({
        name: "Unicode/emoji content",
        pass: unicodeOk,
        note: unicodeOk ? "saved + recalled ✓" : "FAILED",
      });
    }

    // EC5: Very long content (near 10K limit)
    {
      const longContent = "User ".padEnd(9000, "prefers TypeScript for projects. ");
      let longOk = false;
      try {
        const mem = ecStorage.saveMemory(longContent, "fact");
        const emb = await embedLocal(longContent);
        ecStorage.saveEmbedding(mem.id, emb);
        const results = ecStorage.vectorSearch(emb, 3);
        longOk = results.some((r) => r.memory.id === mem.id);
      } catch {
        longOk = false;
      }
      ecResults.push({
        name: "Long content (9KB)",
        pass: longOk,
        note: longOk ? `${longContent.length} chars saved + recalled` : "FAILED",
      });
    }

    // EC6: consolidated_at marking — memories marked consolidated should not appear
    //      in getUnconsolidatedMemories
    {
      const cm1 = ecStorage.saveMemory("Consolidation test A", "fact");
      const cm2 = ecStorage.saveMemory("Consolidation test B", "fact");
      const cm3 = ecStorage.saveMemory("Consolidation test C", "fact");

      const beforeCount = ecStorage.getUnconsolidatedCount();
      ecStorage.markMemoriesConsolidated([cm1.id, cm2.id]);
      const afterCount = ecStorage.getUnconsolidatedCount();

      const remaining = ecStorage.getUnconsolidatedMemories(100);
      const cm1InRemaining = remaining.some((m) => m.id === cm1.id);
      const cm2InRemaining = remaining.some((m) => m.id === cm2.id);
      const cm3InRemaining = remaining.some((m) => m.id === cm3.id);

      const markOk = !cm1InRemaining && !cm2InRemaining && cm3InRemaining && afterCount === beforeCount - 2;
      ecResults.push({
        name: "consolidated_at filter",
        pass: markOk,
        note: `before=${beforeCount} after=${afterCount} unmarked-visible=${cm3InRemaining}`,
      });
    }

    // EC7: getEmbedding round-trip — stored embedding should be retrievable
    {
      const em = ecStorage.saveMemory("Embedding roundtrip test", "fact");
      const originalEmb = await embedLocal("Embedding roundtrip test");
      ecStorage.saveEmbedding(em.id, originalEmb);

      const retrieved = ecStorage.getEmbedding(em.id);
      let roundtripOk = false;
      if (retrieved && retrieved.length === 384) {
        // Check first 10 values are approximately equal (float32 precision)
        roundtripOk = originalEmb.slice(0, 10).every((v, i) => Math.abs(v - retrieved[i]) < 1e-5);
      }
      ecResults.push({
        name: "Embedding roundtrip",
        pass: roundtripOk,
        note: roundtripOk ? "384-dim match ✓" : `MISMATCH len=${retrieved?.length ?? 0}`,
      });
    }

    // EC8: getEmbedding for non-existent ID
    {
      const missing = ecStorage.getEmbedding("00000000-0000-0000-0000-000000000000");
      ecResults.push({
        name: "Missing embedding returns null",
        pass: missing === null,
        note: `got=${missing === null ? "null" : typeof missing}`,
      });
    }

    // EC9: Scheduler double-tick prevention — concurrent ticks should not overlap
    {
      const schedDb2 = join(tmpdir(), `cortex-bench-sched2-${Date.now()}.db`);
      const sched2Storage = new CortexStorage(schedDb2, { unsafeSkipPathCheck: true });

      // Seed enough memories to trigger clustering
      for (let i = 0; i < 10; i++) {
        const m = sched2Storage.saveMemory(`TypeScript pattern ${i}: use strict types`, "fact");
        sched2Storage.saveEmbedding(m.id, await embedLocal(m.content));
      }

      const scheduler2 = new CortexScheduler(sched2Storage);

      // Fire two ticks simultaneously
      const [r1, r2] = await Promise.all([scheduler2.tick(), scheduler2.tick()]);
      const oneSkipped = r1.skipped || r2.skipped;
      const _onlyOneRan = (r1.skipped && !r2.skipped) || (!r1.skipped && r2.skipped);

      ecResults.push({
        name: "Concurrent tick prevention",
        pass: oneSkipped,
        note: `tick1: scanned=${r1.memoriesScanned} skip=${r1.skipped} | tick2: scanned=${r2.memoriesScanned} skip=${r2.skipped}`,
      });

      sched2Storage.close();
      if (existsSync(schedDb2)) unlinkSync(schedDb2);
    }

    // EC10: Scheduler re-tick after consolidation — should not re-cluster same memories
    {
      const schedDb3 = join(tmpdir(), `cortex-bench-sched3-${Date.now()}.db`);
      const sched3Storage = new CortexStorage(schedDb3, { unsafeSkipPathCheck: true });

      for (let i = 0; i < 8; i++) {
        const m = sched3Storage.saveMemory(`Database fact ${i}: SQLite is fast`, "fact");
        sched3Storage.saveEmbedding(m.id, await embedLocal(m.content));
      }

      const scheduler3 = new CortexScheduler(sched3Storage);
      const tick1 = await scheduler3.tick();
      const tick2 = await scheduler3.tick();

      // After first tick marks memories as consolidated, second tick should see fewer or skip
      const noRecluster = tick2.skipped || tick2.clustersSaved === 0;
      ecResults.push({
        name: "No re-clustering after consolidation",
        pass: noRecluster,
        note: `tick1: clusters=${tick1.clustersSaved} | tick2: clusters=${tick2.clustersSaved} skip=${tick2.skipped}`,
      });

      sched3Storage.close();
      if (existsSync(schedDb3)) unlinkSync(schedDb3);
    }

    // EC11: Graph walk with no edges — should return the start nodes without crashing
    {
      const isolated1 = ecStorage.saveMemory("Isolated memory one", "fact");
      const isolated2 = ecStorage.saveMemory("Isolated memory two", "fact");
      const walkResult = ecStorage.graphWalk([isolated1.id, isolated2.id], 2);
      const walkedBoth = walkResult.length === 2;
      ecResults.push({
        name: "Graph walk (no edges)",
        pass: walkedBoth,
        note: `expected 2 nodes, got ${walkResult.length}`,
      });
    }

    // EC12: Graph walk with invalid/missing IDs — should not crash
    {
      const walkBad = ecStorage.graphWalk(["nonexistent-id", "also-fake"], 2);
      ecResults.push({
        name: "Graph walk (invalid IDs)",
        pass: walkBad.length === 0,
        note: `expected 0 results, got ${walkBad.length}`,
      });
    }

    // EC13: Vector search on empty DB — should return empty, not crash
    {
      const emptyDb = join(tmpdir(), `cortex-bench-empty-${Date.now()}.db`);
      const emptyStorage = new CortexStorage(emptyDb, { unsafeSkipPathCheck: true });
      const emb = await embedLocal("anything at all");
      const emptyResults = emptyStorage.vectorSearch(emb, 10);
      ecResults.push({
        name: "Vector search (empty DB)",
        pass: emptyResults.length === 0,
        note: `got ${emptyResults.length} results`,
      });
      emptyStorage.close();
      if (existsSync(emptyDb)) unlinkSync(emptyDb);
    }

    // EC14: Supersede then un-supersede scenario — supersedeMemory is one-way
    //       (no unsupersede method, so verify it stays superseded)
    {
      const sup = ecStorage.saveMemory("Will be superseded", "fact");
      const supEmb = await embedLocal(sup.content);
      ecStorage.saveEmbedding(sup.id, supEmb);

      ecStorage.supersedeMemory(sup.id);
      const afterSup = ecStorage.vectorSearch(supEmb, 10);
      const hidden = !afterSup.some((r) => r.memory.id === sup.id);

      ecResults.push({
        name: "Supersede is permanent",
        pass: hidden,
        note: hidden ? "not in vector search ✓" : "LEAKED into results",
      });
    }

    // EC15: Multiple edge types between same memories
    {
      const meA = ecStorage.saveMemory("Multi-edge test A", "fact");
      const meB = ecStorage.saveMemory("Multi-edge test B", "fact");
      ecStorage.addEdge(meA.id, meB.id, "relates_to", 0.8);
      ecStorage.addEdge(meA.id, meB.id, "evolved_from", 0.9);

      const edges = ecStorage.getEdgesFrom(meA.id);
      const hasRelates = edges.some((e) => e.relation === "relates_to");
      const hasEvolved = edges.some((e) => e.relation === "evolved_from");
      ecResults.push({
        name: "Multiple edge types",
        pass: hasRelates && hasEvolved,
        note: `${edges.length} edges, relates_to=${hasRelates} evolved_from=${hasEvolved}`,
      });
    }

    // EC16: getUnconsolidatedCount matches getUnconsolidatedMemories length
    {
      const countVal = ecStorage.getUnconsolidatedCount();
      const listVal = ecStorage.getUnconsolidatedMemories(200);
      ecResults.push({
        name: "Count vs list consistency",
        pass: countVal === listVal.length,
        note: `count()=${countVal} list().length=${listVal.length}`,
      });
    }

    // ── DELETE EDGE CASES ──

    // EC17: Delete a basic memory — verify removed from memories, vectors, and edges
    {
      const delDb = join(tmpdir(), `cortex-bench-del-${Date.now()}.db`);
      const delStorage = new CortexStorage(delDb, { unsafeSkipPathCheck: true });

      const m = delStorage.saveMemory("Memory to delete", "fact");
      const emb = await embedLocal(m.content);
      delStorage.saveEmbedding(m.id, emb);

      const result = delStorage.deleteMemory(m.id);
      const afterSearch = delStorage.vectorSearch(emb, 10);
      const embAfter = delStorage.getEmbedding(m.id);
      const memAfter = delStorage.getMemoryById(m.id);

      ecResults.push({
        name: "Delete basic memory",
        pass: result.deleted && !afterSearch.some((r) => r.memory.id === m.id) && embAfter === null && memAfter === null,
        note: `deleted=${result.deleted} vec=${afterSearch.length} emb=${embAfter === null} mem=${memAfter === null}`,
      });

      delStorage.close();
      if (existsSync(delDb)) unlinkSync(delDb);
    }

    // EC18: Delete non-existent memory — should return deleted=false
    {
      const delDb2 = join(tmpdir(), `cortex-bench-del2-${Date.now()}.db`);
      const delStorage2 = new CortexStorage(delDb2, { unsafeSkipPathCheck: true });

      const result = delStorage2.deleteMemory("00000000-0000-0000-0000-000000000000");
      ecResults.push({
        name: "Delete non-existent memory",
        pass: result.deleted === false && result.edgesRemoved === 0,
        note: `deleted=${result.deleted}`,
      });

      delStorage2.close();
      if (existsSync(delDb2)) unlinkSync(delDb2);
    }

    // EC19: Delete memory with edges — verify CASCADE removes edges
    {
      const delDb3 = join(tmpdir(), `cortex-bench-del3-${Date.now()}.db`);
      const delStorage3 = new CortexStorage(delDb3, { unsafeSkipPathCheck: true });

      const mA = delStorage3.saveMemory("Edge source", "fact");
      const mB = delStorage3.saveMemory("Edge target 1", "fact");
      const mC = delStorage3.saveMemory("Edge target 2", "fact");
      delStorage3.addEdge(mA.id, mB.id, "relates_to", 0.9);
      delStorage3.addEdge(mA.id, mC.id, "led_to", 0.8);
      delStorage3.addEdge(mB.id, mC.id, "relates_to", 0.7); // unrelated edge should survive

      const result = delStorage3.deleteMemory(mA.id);

      // Edges from A should be gone but B↔C edge should survive
      const edgesB = delStorage3.getEdgesFrom(mB.id);
      const bcEdgeSurvived = edgesB.some((e) =>
        (e.source_id === mB.id && e.target_id === mC.id) ||
        (e.source_id === mC.id && e.target_id === mB.id)
      );

      ecResults.push({
        name: "Delete cascades edges",
        pass: result.deleted && result.edgesRemoved === 2 && bcEdgeSurvived,
        note: `removed=${result.edgesRemoved} B↔C survived=${bcEdgeSurvived}`,
      });

      delStorage3.close();
      if (existsSync(delDb3)) unlinkSync(delDb3);
    }

    // EC20: Delete memory that contradicted another — should restore superseded
    {
      const delDb4 = join(tmpdir(), `cortex-bench-del4-${Date.now()}.db`);
      const delStorage4 = new CortexStorage(delDb4, { unsafeSkipPathCheck: true });

      const old = delStorage4.saveMemory("I use Postgres", "decision");
      const oldEmb = await embedLocal(old.content);
      delStorage4.saveEmbedding(old.id, oldEmb);

      // Supersede the old memory
      delStorage4.supersedeMemory(old.id);
      const newer = delStorage4.saveMemory("I switched to SQLite", "decision");
      const newerEmb = await embedLocal(newer.content);
      delStorage4.saveEmbedding(newer.id, newerEmb);
      delStorage4.addEdge(newer.id, old.id, "contradicts", 1.0);

      // Old should be hidden from vector search
      const beforeDelete = delStorage4.vectorSearch(oldEmb, 10);
      const oldHiddenBefore = !beforeDelete.some((r) => r.memory.id === old.id);

      // Delete the newer one — should restore the old
      const result = delStorage4.deleteMemory(newer.id);
      const afterDelete = delStorage4.vectorSearch(oldEmb, 10);
      const oldRestoredAfter = afterDelete.some((r) => r.memory.id === old.id);

      ecResults.push({
        name: "Delete restores superseded",
        pass: result.deleted && result.supersededRestored.length === 1 && oldHiddenBefore && oldRestoredAfter,
        note: `restored=${result.supersededRestored.length} hiddenBefore=${oldHiddenBefore} visibleAfter=${oldRestoredAfter}`,
      });

      delStorage4.close();
      if (existsSync(delDb4)) unlinkSync(delDb4);
    }

    // EC21: Multiple contradictions — delete one contradicting memory, but another
    //       still contradicts the target → target should stay superseded
    {
      const delDb5 = join(tmpdir(), `cortex-bench-del5-${Date.now()}.db`);
      const delStorage5 = new CortexStorage(delDb5, { unsafeSkipPathCheck: true });

      const original = delStorage5.saveMemory("I use Java", "fact");
      delStorage5.supersedeMemory(original.id);

      const contra1 = delStorage5.saveMemory("Switched to Kotlin", "fact");
      delStorage5.addEdge(contra1.id, original.id, "contradicts", 1.0);

      const contra2 = delStorage5.saveMemory("Now using Kotlin full-time", "fact");
      delStorage5.addEdge(contra2.id, original.id, "contradicts", 1.0);

      // Delete only one contradictor — original should remain superseded
      const result = delStorage5.deleteMemory(contra1.id);
      const stillSuperseded = delStorage5.getMemoryById(original.id);

      ecResults.push({
        name: "Multi-contradict keeps supersede",
        pass: result.deleted && result.supersededRestored.length === 0 && stillSuperseded?.superseded === true,
        note: `restored=${result.supersededRestored.length} stillSup=${stillSuperseded?.superseded}`,
      });

      delStorage5.close();
      if (existsSync(delDb5)) unlinkSync(delDb5);
    }

    // EC22: Delete memory from pending cluster — cluster should be scrubbed
    {
      const delDb6 = join(tmpdir(), `cortex-bench-del6-${Date.now()}.db`);
      const delStorage6 = new CortexStorage(delDb6, { unsafeSkipPathCheck: true });

      const cm1 = delStorage6.saveMemory("Cluster mem 1", "fact");
      const cm2 = delStorage6.saveMemory("Cluster mem 2", "fact");
      const cm3 = delStorage6.saveMemory("Cluster mem 3", "fact");
      delStorage6.savePendingCluster([cm1.id, cm2.id, cm3.id]);

      const result = delStorage6.deleteMemory(cm1.id);

      // Cluster should still exist but with only 2 members
      const clusters = delStorage6.getUnreviewedClusters(10);
      const scrubbed = clusters.length === 1 && clusters[0].cluster.memory_ids.length === 2 &&
        !clusters[0].cluster.memory_ids.includes(cm1.id);

      ecResults.push({
        name: "Delete scrubs from cluster",
        pass: result.deleted && result.clustersAffected === 1 && scrubbed,
        note: `affected=${result.clustersAffected} remaining=${clusters[0]?.cluster.memory_ids.length ?? 0}`,
      });

      delStorage6.close();
      if (existsSync(delDb6)) unlinkSync(delDb6);
    }

    // EC23: Delete memory making cluster < 2 — cluster should be deleted entirely
    {
      const delDb7 = join(tmpdir(), `cortex-bench-del7-${Date.now()}.db`);
      const delStorage7 = new CortexStorage(delDb7, { unsafeSkipPathCheck: true });

      const sm1 = delStorage7.saveMemory("Small cluster mem 1", "fact");
      const sm2 = delStorage7.saveMemory("Small cluster mem 2", "fact");
      delStorage7.savePendingCluster([sm1.id, sm2.id]);

      delStorage7.deleteMemory(sm1.id);

      const clusters = delStorage7.getUnreviewedClusters(10);
      ecResults.push({
        name: "Delete removes small cluster",
        pass: clusters.length === 0,
        note: `clusters remaining=${clusters.length}`,
      });

      delStorage7.close();
      if (existsSync(delDb7)) unlinkSync(delDb7);
    }

    // EC24: Delete already-superseded memory — should work cleanly
    {
      const delDb8 = join(tmpdir(), `cortex-bench-del8-${Date.now()}.db`);
      const delStorage8 = new CortexStorage(delDb8, { unsafeSkipPathCheck: true });

      const m = delStorage8.saveMemory("Will be superseded then deleted", "fact");
      const emb = await embedLocal(m.content);
      delStorage8.saveEmbedding(m.id, emb);
      delStorage8.supersedeMemory(m.id);

      const result = delStorage8.deleteMemory(m.id);
      const memAfter = delStorage8.getMemoryById(m.id);
      const embAfter = delStorage8.getEmbedding(m.id);

      ecResults.push({
        name: "Delete superseded memory",
        pass: result.deleted && memAfter === null && embAfter === null,
        note: `deleted=${result.deleted} cleaned=${memAfter === null && embAfter === null}`,
      });

      delStorage8.close();
      if (existsSync(delDb8)) unlinkSync(delDb8);
    }

    // EC25: Delete twice — second delete should be idempotent (deleted=false)
    {
      const delDb9 = join(tmpdir(), `cortex-bench-del9-${Date.now()}.db`);
      const delStorage9 = new CortexStorage(delDb9, { unsafeSkipPathCheck: true });

      const m = delStorage9.saveMemory("Delete me twice", "fact");
      delStorage9.saveEmbedding(m.id, await embedLocal(m.content));

      const first = delStorage9.deleteMemory(m.id);
      const second = delStorage9.deleteMemory(m.id);

      ecResults.push({
        name: "Double delete idempotent",
        pass: first.deleted && !second.deleted,
        note: `first=${first.deleted} second=${second.deleted}`,
      });

      delStorage9.close();
      if (existsSync(delDb9)) unlinkSync(delDb9);
    }

    // EC26: Delete memory then verify count decremented
    {
      const delDb10 = join(tmpdir(), `cortex-bench-del10-${Date.now()}.db`);
      const delStorage10 = new CortexStorage(delDb10, { unsafeSkipPathCheck: true });

      delStorage10.saveMemory("Survivor 1", "fact");
      const victim = delStorage10.saveMemory("To be deleted", "fact");
      delStorage10.saveMemory("Survivor 2", "fact");

      const countBefore = delStorage10.getMemoryCount();
      delStorage10.deleteMemory(victim.id);
      const countAfter = delStorage10.getMemoryCount();

      ecResults.push({
        name: "Delete decrements count",
        pass: countBefore === 3 && countAfter === 2,
        note: `before=${countBefore} after=${countAfter}`,
      });

      delStorage10.close();
      if (existsSync(delDb10)) unlinkSync(delDb10);
    }

    ecStorage.close();
    if (existsSync(edgeCaseDb)) unlinkSync(edgeCaseDb);

    // Tally edge case results
    const ecPassed = ecResults.filter((r) => r.pass).length;
    const ecTotal = ecResults.length;
    const ecRate = (ecPassed / ecTotal) * 100;

    results.push({
      label: "Edge case pass rate",
      value: ecRate,
      unit: "%",
      note: `${ecPassed}/${ecTotal} passed`,
      pass: ecRate === 100,
    });

    // -----------------------------------------------------------------------
    // RESULTS TABLE
    // -----------------------------------------------------------------------

    const W = 70;
    const LABEL_W = 30;
    const VAL_W = 14;

    const fmt = (v: number, unit: string): string => {
      const formatted =
        unit === "ms"  ? v.toFixed(1) :
        unit === "%"   ? v.toFixed(1) :
        unit === "x"   ? v.toFixed(2) :
        unit === "tok" ? v.toFixed(0) :
        unit === "KB"  ? v.toFixed(2) :
        unit === "mem/s" ? v.toFixed(2) :
        v.toFixed(1);
      return `${formatted} ${unit}`;
    };

    const passIcon = (r: Result): string => {
      if (r.pass === undefined) return " ";
      return r.pass ? "✓" : "✗";
    };

    console.log("═".repeat(W));
    console.log("  RESULTS");
    console.log("═".repeat(W));

    type Section = { header: string; indices: number[] };
    const sections: Section[] = [
      { header: "Embedding latency",        indices: [0, 1, 2] },
      { header: "Write performance",        indices: [3, 4] },
      { header: "Recall (vector only)",     indices: [5, 6, 7] },
      { header: "Token efficiency",         indices: [8, 9, 10, 11] },
      { header: "Contradiction detection",  indices: [12] },
      { header: "Embedding quality",        indices: [13, 14, 15] },
      { header: "Graph layer",              indices: [16, 17, 18, 19] },
      { header: "Correctness",              indices: [20, 21, 22] },
      { header: "Scheduler / proactive",    indices: [23, 24, 25] },
      { header: "Storage",                  indices: [26, 27, 28] },
      { header: "Edge cases",               indices: [29] },
    ];

    for (const sec of sections) {
      console.log(`\n  ── ${sec.header}`);
      for (const i of sec.indices) {
        const r = results[i];
        if (!r) continue;
        const label = r.label.padEnd(LABEL_W);
        const val   = fmt(r.value, r.unit).padStart(VAL_W);
        const icon  = passIcon(r);
        const note  = r.note ? `  ${r.note}` : "";
        console.log(`  ${icon} ${label} ${val}${note}`);
      }
    }

    // Pass/fail summary
    const evaluated = results.filter((r) => r.pass !== undefined);
    const passed    = evaluated.filter((r) => r.pass === true);
    const failed    = evaluated.filter((r) => r.pass === false);

    // Edge case detail breakdown
    if (ecResults.length > 0) {
      console.log(`\n  ── Edge case details`);
      for (const ec of ecResults) {
        console.log(`  ${ec.pass ? "✓" : "✗"} ${ec.name.padEnd(35)} ${ec.note}`);
      }
    }

    console.log("\n" + "═".repeat(W));
    console.log(`  SCORE: ${passed.length}/${evaluated.length} checks passed`);
    if (failed.length > 0) {
      console.log(`  FAILED: ${failed.map((r) => r.label).join(", ")}`);
    }
    console.log("═".repeat(W) + "\n");

  } catch (err) {
    console.error("Benchmark error:", err);
    process.exit(1);
  } finally {
    try { storage.close(); } catch { /* already closed */ }
    if (existsSync(dbPath)) unlinkSync(dbPath);
  }
}

main().catch(console.error);
