/**
 * CortexScheduler — background proactive agent
 *
 * Runs on a timer (default: every 30 minutes) and:
 *   1. Pulls unconsolidated explicit memories from the DB
 *   2. Computes embeddings and clusters them by cosine similarity
 *   3. Saves clusters with 2+ members as `pending_clusters`
 *
 * Claude surfaces and synthesises these via `memory_proactive`.
 * This makes Cortex proactive — it continuously organises knowledge
 * in the background without waiting for a user prompt.
 */

import { embedLocal } from "../embeddings/engine.js";
import type { CortexStorage } from "../storage/database.js";

export type SchedulerLogFn = (message: string) => void;

function cosineSim(a: number[], b: number[]): number {
  let dot = 0, magA = 0, magB = 0;
  for (let i = 0; i < a.length; i++) {
    dot  += a[i] * b[i];
    magA += a[i] * a[i];
    magB += b[i] * b[i];
  }
  const denom = Math.sqrt(magA) * Math.sqrt(magB);
  return denom === 0 ? 0 : dot / denom;
}

export interface TickResult {
  memoriesScanned: number;
  clustersFound:   number;
  clustersSaved:   number;
  skipped:         boolean;
  reason?:         string;
}

export class CortexScheduler {
  private storage:  CortexStorage;
  private timer:    ReturnType<typeof setInterval> | null = null;
  private running:  boolean = false;
  private log:      SchedulerLogFn;

  /** Tick in progress — prevents overlapping runs */
  private ticking:  boolean = false;

  constructor(storage: CortexStorage, log?: SchedulerLogFn) {
    this.storage = storage;
    this.log = log ?? (() => {});
  }

  /**
   * Start the scheduler. Runs an initial tick after `initialDelayMs`
   * (default 10 s to let the MCP server finish startup), then repeats
   * every `intervalMs`.
   */
  start(intervalMs: number = 30 * 60 * 1000, initialDelayMs: number = 10_000): void {
    if (this.running) return;
    this.running = true;

    this.log(`[cortex] Scheduler started — ticking every ${Math.round(intervalMs / 60_000)} min`);

    // Initial tick after a short delay so MCP startup finishes first
    setTimeout(async () => {
      if (this.running) await this.tick();
    }, initialDelayMs);

    this.timer = setInterval(async () => {
      if (this.running) await this.tick();
    }, intervalMs);
  }

  stop(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer  = null;
    this.running = false;
    this.log("[cortex] Scheduler stopped");
  }

  get isRunning(): boolean {
    return this.running;
  }

  /**
   * Run one background consolidation pass.
   * Safe to call manually (e.g., from tests or `cortex daemon --once`).
   */
  async tick(): Promise<TickResult> {
    if (this.ticking) {
      return { memoriesScanned: 0, clustersFound: 0, clustersSaved: 0, skipped: true, reason: "previous tick still running" };
    }
    this.ticking = true;

    try {
      const memories = this.storage.getUnconsolidatedMemories(50);

      if (memories.length < 5) {
        this.log(`[cortex] Scheduler tick — only ${memories.length} memories, waiting for more`);
        return { memoriesScanned: memories.length, clustersFound: 0, clustersSaved: 0, skipped: true, reason: `${memories.length} memories < 5 threshold` };
      }

      this.log(`[cortex] Scheduler tick — scanning ${memories.length} memories...`);

      // Compute embeddings for every candidate memory
      const embeddings = new Map<string, number[]>();
      for (const m of memories) {
        embeddings.set(m.id, await embedLocal(m.content));
      }

      // Greedy cosine-similarity clustering (threshold: 0.70)
      const clustered = new Set<string>();
      const clusters:  string[][] = [];

      for (const m of memories) {
        if (clustered.has(m.id)) continue;
        const anchor  = embeddings.get(m.id)!;
        const cluster = [m.id];
        clustered.add(m.id);

        for (const other of memories) {
          if (clustered.has(other.id) || cluster.length >= 8) continue;
          const sim = cosineSim(anchor, embeddings.get(other.id)!);
          if (sim >= 0.70) {
            cluster.push(other.id);
            clustered.add(other.id);
          }
        }

        if (cluster.length >= 2) clusters.push(cluster);
      }

      if (clusters.length === 0) {
        this.log("[cortex] Scheduler tick — memories are too diverse to cluster yet");
        return { memoriesScanned: memories.length, clustersFound: 0, clustersSaved: 0, skipped: false };
      }

      // Persist clusters so Claude can review them later via memory_proactive
      let saved = 0;
      for (const cluster of clusters) {
        this.storage.savePendingCluster(cluster);
        saved++;
      }

      this.storage.setLastConsolidationTime(new Date().toISOString());

      this.log(`[cortex] Scheduler tick — saved ${saved} pending cluster(s) from ${memories.length} memories`);

      return {
        memoriesScanned: memories.length,
        clustersFound:   clusters.length,
        clustersSaved:   saved,
        skipped:         false,
      };

    } finally {
      this.ticking = false;
    }
  }
}
