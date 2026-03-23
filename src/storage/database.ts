import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import { randomUUID } from "crypto";
import { dirname, resolve as pathResolve, sep } from "path";
import { mkdirSync, existsSync } from "fs";
import { homedir } from "os";
import type { Memory, Edge, MemoryCategory, MemorySource, EdgeRelation, PendingCluster } from "../types/index.js";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export class CortexStorage {
  private db: Database.Database;

  constructor(dbPath: string, opts?: { unsafeSkipPathCheck?: boolean }) {
    const resolved = pathResolve(dbPath.replace(/^~/, homedir()));
    if (!opts?.unsafeSkipPathCheck) {
      const allowedBase = pathResolve(homedir(), ".cortex");
      if (!resolved.startsWith(allowedBase + sep) && resolved !== allowedBase) {
        throw new Error(`dbPath must be within ${allowedBase}`);
      }
    }
    const dir = dirname(resolved);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    this.db = new Database(resolved);
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("foreign_keys = ON");

    sqliteVec.load(this.db);
    this.migrate();
  }

  private migrate(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS memories (
        id              TEXT PRIMARY KEY,
        content         TEXT NOT NULL,
        category        TEXT NOT NULL DEFAULT 'fact',
        source          TEXT NOT NULL DEFAULT 'explicit',
        confidence      REAL DEFAULT 1.0,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL,
        superseded      INTEGER DEFAULT 0,
        consolidated_at TEXT
      );

      CREATE TABLE IF NOT EXISTS edges (
        id          TEXT PRIMARY KEY,
        source_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        target_id   TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        relation    TEXT NOT NULL,
        strength    REAL DEFAULT 1.0,
        created_at  TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS consolidation_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
      );

      CREATE TABLE IF NOT EXISTS pending_clusters (
        id          TEXT PRIMARY KEY,
        memory_ids  TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        reviewed    INTEGER DEFAULT 0
      );

      CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
      CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
      CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
      CREATE INDEX IF NOT EXISTS idx_memories_superseded ON memories(superseded);

      CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
        memory_id TEXT PRIMARY KEY,
        embedding FLOAT[384]
      );
    `);
  }

  saveMemory(
    content: string,
    category: MemoryCategory = "fact",
    source: MemorySource = "explicit",
    confidence: number = 1.0
  ): Memory {
    const id = randomUUID();
    const now = new Date().toISOString();

    this.db
      .prepare(
        `INSERT INTO memories (id, content, category, source, confidence, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)`
      )
      .run(id, content, category, source, confidence, now, now);

    return { id, content, category, source, confidence, created_at: now, updated_at: now, superseded: false };
  }

  saveEmbedding(memoryId: string, embedding: number[]): void {
    this.db
      .prepare(`INSERT INTO memory_vectors (memory_id, embedding) VALUES (?, ?)`)
      .run(memoryId, new Float32Array(embedding));
  }

  vectorSearch(queryEmbedding: number[], topK: number = 10): Array<{ memory: Memory; distance: number }> {
    const results = this.db
      .prepare(
        `SELECT
           m.id, m.content, m.category, m.source, m.confidence,
           m.created_at, m.updated_at, m.superseded,
           vec_distance_cosine(v.embedding, ?) as distance
         FROM memory_vectors v
         INNER JOIN memories m ON m.id = v.memory_id
         WHERE m.superseded = 0
         ORDER BY distance ASC
         LIMIT ?`
      )
      .all(new Float32Array(queryEmbedding), topK) as Array<any>;

    return results.map((row) => ({
      memory: {
        id: row.id,
        content: row.content,
        category: row.category as MemoryCategory,
        source: row.source as MemorySource,
        confidence: row.confidence,
        created_at: row.created_at,
        updated_at: row.updated_at,
        superseded: !!row.superseded,
      },
      distance: row.distance,
    }));
  }

  edgeExists(sourceId: string, targetId: string): boolean {
    const row = this.db
      .prepare(
        `SELECT 1 FROM edges WHERE (source_id = ? AND target_id = ?) OR (source_id = ? AND target_id = ?) LIMIT 1`
      )
      .get(sourceId, targetId, targetId, sourceId);
    return !!row;
  }

  addEdge(sourceId: string, targetId: string, relation: EdgeRelation, strength: number = 1.0): Edge {
    const id = randomUUID();
    const now = new Date().toISOString();

    this.db
      .prepare(
        `INSERT INTO edges (id, source_id, target_id, relation, strength, created_at)
         VALUES (?, ?, ?, ?, ?, ?)`
      )
      .run(id, sourceId, targetId, relation, strength, now);

    return { id, source_id: sourceId, target_id: targetId, relation, strength, created_at: now };
  }

  getEdgesFrom(memoryId: string): Edge[] {
    return this.db
      .prepare(`SELECT * FROM edges WHERE source_id = ? OR target_id = ?`)
      .all(memoryId, memoryId) as Edge[];
  }

  graphWalk(startIds: string[], maxHops: number = 2): Array<{ memory: Memory; edges: Edge[]; depth: number }> {
    const visited = new Set<string>();
    const results: Array<{ memory: Memory; edges: Edge[]; depth: number }> = [];

    let currentIds = startIds.filter((id) => !visited.has(id));

    for (let depth = 0; depth <= maxHops && currentIds.length > 0; depth++) {
      const unvisited = currentIds.filter((id) => !visited.has(id));
      if (unvisited.length === 0) break;
      for (const id of unvisited) visited.add(id);

      // Batch-fetch memories for this depth level
      const placeholders = unvisited.map(() => "?").join(",");
      const memRows = this.db
        .prepare(`SELECT * FROM memories WHERE id IN (${placeholders}) AND superseded = 0`)
        .all(...unvisited) as any[];
      const memById = new Map(memRows.map((m) => [m.id, m]));

      // Batch-fetch edges for this depth level
      const edgeRows = this.db
        .prepare(`SELECT * FROM edges WHERE source_id IN (${placeholders}) OR target_id IN (${placeholders})`)
        .all(...unvisited, ...unvisited) as Edge[];
      const edgesByNode = new Map<string, Edge[]>();
      for (const id of unvisited) edgesByNode.set(id, []);
      for (const e of edgeRows) {
        edgesByNode.get(e.source_id)?.push(e);
        if (e.source_id !== e.target_id) edgesByNode.get(e.target_id)?.push(e);
      }

      const nextIds: string[] = [];
      for (const id of unvisited) {
        const memory = memById.get(id);
        if (!memory) continue;

        const edges = edgesByNode.get(id) ?? [];
        results.push({
          memory: {
            id: memory.id,
            content: memory.content,
            category: memory.category,
            source: memory.source,
            confidence: memory.confidence,
            created_at: memory.created_at,
            updated_at: memory.updated_at,
            superseded: !!memory.superseded,
          },
          edges,
          depth,
        });

        for (const e of edges) {
          const neighbor = e.source_id === id ? e.target_id : e.source_id;
          if (!visited.has(neighbor)) nextIds.push(neighbor);
        }
      }

      currentIds = nextIds;
    }

    return results;
  }

  findContradictions(): Memory[] {
    return this.db
      .prepare(
        `SELECT DISTINCT m.* FROM memories m
         INNER JOIN edges e ON (e.source_id = m.id OR e.target_id = m.id)
         WHERE e.relation = 'contradicts' AND m.superseded = 0`
      )
      .all() as Memory[];
  }

  supersedeMemory(memoryId: string): void {
    const now = new Date().toISOString();
    this.db
      .prepare(`UPDATE memories SET superseded = 1, updated_at = ? WHERE id = ?`)
      .run(now, memoryId);
  }

  getRecentMemories(limit: number = 20): Memory[] {
    return this.db
      .prepare(`SELECT * FROM memories WHERE superseded = 0 ORDER BY created_at DESC LIMIT ?`)
      .all(limit) as Memory[];
  }

  getAllMemories(): Memory[] {
    return this.db
      .prepare(`SELECT * FROM memories WHERE superseded = 0 ORDER BY updated_at DESC`)
      .all() as Memory[];
  }

  getMemoryCount(): number {
    const row = this.db.prepare(`SELECT COUNT(*) as count FROM memories WHERE superseded = 0`).get() as any;
    return row.count;
  }

  /**
   * Returns raw memories for consolidation — explicit facts/preferences/decisions
   * that haven't been consolidated into insights yet, oldest first.
   */
  getUnconsolidatedMemories(limit: number = 30): Memory[] {
    return this.db
      .prepare(
        `SELECT * FROM memories
         WHERE superseded = 0
           AND source = 'explicit'
           AND category != 'insight'
           AND consolidated_at IS NULL
         ORDER BY created_at ASC
         LIMIT ?`
      )
      .all(limit) as Memory[];
  }

  getUnconsolidatedCount(): number {
    const row = this.db
      .prepare(
        `SELECT COUNT(*) as count FROM memories
         WHERE superseded = 0
           AND source = 'explicit'
           AND category != 'insight'
           AND consolidated_at IS NULL`
      )
      .get() as any;
    return row.count;
  }

  markMemoriesConsolidated(ids: string[]): void {
    if (ids.length === 0) return;
    const now = new Date().toISOString();
    const placeholders = ids.map(() => "?").join(",");
    this.db
      .prepare(`UPDATE memories SET consolidated_at = ? WHERE id IN (${placeholders})`)
      .run(now, ...ids);
  }

  /**
   * Returns all memories related to a query embedding, sorted chronologically,
   * with their outgoing edges — used to build a timeline of how a topic evolved.
   */
  getTimelineMemories(queryEmbedding: number[], topK: number = 15): Array<{ memory: Memory; edges: Edge[]; distance: number }> {
    const results = this.db
      .prepare(
        `SELECT
           m.id, m.content, m.category, m.source, m.confidence,
           m.created_at, m.updated_at, m.superseded,
           vec_distance_cosine(v.embedding, ?) as distance
         FROM memory_vectors v
         INNER JOIN memories m ON m.id = v.memory_id
         ORDER BY distance ASC
         LIMIT ?`
      )
      .all(new Float32Array(queryEmbedding), topK) as Array<any>;

    return results
      .map((row) => ({
        memory: {
          id: row.id,
          content: row.content,
          category: row.category as MemoryCategory,
          source: row.source as MemorySource,
          confidence: row.confidence,
          created_at: row.created_at,
          updated_at: row.updated_at,
          superseded: !!row.superseded,
        },
        edges: this.getEdgesFrom(row.id),
        distance: row.distance,
      }))
      .sort((a, b) => a.memory.created_at.localeCompare(b.memory.created_at));
  }

  getLastConsolidationTime(): string | null {
    const row = this.db
      .prepare(`SELECT value FROM consolidation_meta WHERE key = 'last_run'`)
      .get() as any;
    return row?.value ?? null;
  }

  setLastConsolidationTime(isoTime: string): void {
    this.db
      .prepare(
        `INSERT INTO consolidation_meta (key, value) VALUES ('last_run', ?)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value`
      )
      .run(isoTime);
  }

  savePendingCluster(memoryIds: string[]): PendingCluster {
    const id = randomUUID();
    const now = new Date().toISOString();
    const json = JSON.stringify(memoryIds);
    this.db
      .prepare(`INSERT INTO pending_clusters (id, memory_ids, created_at) VALUES (?, ?, ?)`)
      .run(id, json, now);
    return { id, memory_ids: memoryIds, created_at: now, reviewed: false };
  }

  getUnreviewedClusters(limit: number = 10): Array<{ cluster: PendingCluster; memories: Memory[] }> {
    const rows = this.db
      .prepare(`SELECT * FROM pending_clusters WHERE reviewed = 0 ORDER BY created_at ASC LIMIT ?`)
      .all(limit) as Array<{ id: string; memory_ids: string; created_at: string; reviewed: number }>;

    return rows.map((row) => {
      const parsed = JSON.parse(row.memory_ids);
      const ids: string[] = (Array.isArray(parsed) ? parsed : [])
        .filter((v: unknown): v is string => typeof v === "string" && UUID_RE.test(v))
        .slice(0, 100);
      // Batch-fetch all memories for this cluster in one query
      const placeholders = ids.map(() => "?").join(",");
      const memRows = ids.length > 0
        ? (this.db.prepare(`SELECT * FROM memories WHERE id IN (${placeholders})`).all(...ids) as any[])
        : [];
      const byId = new Map(memRows.map((m) => [m.id, m]));
      const memories = ids
        .map((mid) => byId.get(mid))
        .filter(Boolean)
        .map((m) => ({
          id: m.id,
          content: m.content,
          category: m.category as MemoryCategory,
          source: m.source as MemorySource,
          confidence: m.confidence,
          created_at: m.created_at,
          updated_at: m.updated_at,
          superseded: !!m.superseded,
        }));
      return {
        cluster: { id: row.id, memory_ids: ids, created_at: row.created_at, reviewed: !!row.reviewed },
        memories,
      };
    });
  }

  markClusterReviewed(id: string): void {
    this.db.prepare(`UPDATE pending_clusters SET reviewed = 1 WHERE id = ?`).run(id);
  }

  getPendingClusterCount(): number {
    const row = this.db
      .prepare(`SELECT COUNT(*) as count FROM pending_clusters WHERE reviewed = 0`)
      .get() as any;
    return row.count;
  }

  getMemoryById(id: string): Memory | null {
    const row = this.db
      .prepare(`SELECT * FROM memories WHERE id = ?`)
      .get(id) as any;
    if (!row) return null;
    return {
      id: row.id,
      content: row.content,
      category: row.category as MemoryCategory,
      source: row.source as MemorySource,
      confidence: row.confidence,
      created_at: row.created_at,
      updated_at: row.updated_at,
      superseded: !!row.superseded,
    };
  }

  /**
   * Delete a memory and clean up all references.
   *
   * Handles:
   * 1. edges           — CASCADE from FK, auto-deleted
   * 2. memory_vectors  — vec0 virtual table, manual DELETE (no CASCADE support)
   * 3. pending_clusters — scrub ID from JSON; delete cluster if < 2 members remain
   * 4. supersede chain — if this memory contradicted others, restore them
   *                      (only if no OTHER memory also contradicts the target)
   *
   * Everything runs in a single transaction for atomicity.
   */
  deleteMemory(id: string): {
    deleted: boolean;
    edgesRemoved: number;
    supersededRestored: string[];
    clustersAffected: number;
  } {
    const txn = this.db.transaction(() => {
      // 1. Check existence
      const memory = this.db.prepare(`SELECT id FROM memories WHERE id = ?`).get(id) as any;
      if (!memory) {
        return { deleted: false, edgesRemoved: 0, supersededRestored: [], clustersAffected: 0 };
      }

      // 2. Count edges that will be cascade-deleted (for reporting)
      const edgeRow = this.db
        .prepare(`SELECT COUNT(*) as count FROM edges WHERE source_id = ? OR target_id = ?`)
        .get(id, id) as any;
      const edgesRemoved: number = edgeRow.count;

      // 3. Supersede chain restoration:
      //    Find memories that THIS memory contradicted (source → target with relation='contradicts')
      //    Only un-supersede if no OTHER memory also contradicts that target
      const contradictedTargets = this.db
        .prepare(`SELECT target_id FROM edges WHERE source_id = ? AND relation = 'contradicts'`)
        .all(id) as Array<{ target_id: string }>;

      const supersededRestored: string[] = [];
      const now = new Date().toISOString();

      for (const { target_id } of contradictedTargets) {
        const otherRow = this.db
          .prepare(
            `SELECT COUNT(*) as count FROM edges
             WHERE target_id = ? AND relation = 'contradicts' AND source_id != ?`
          )
          .get(target_id, id) as any;

        if (otherRow.count === 0) {
          this.db
            .prepare(`UPDATE memories SET superseded = 0, updated_at = ? WHERE id = ?`)
            .run(now, target_id);
          supersededRestored.push(target_id);
        }
      }

      // 4. Delete from memory_vectors (vec0 virtual table — no CASCADE)
      this.db.prepare(`DELETE FROM memory_vectors WHERE memory_id = ?`).run(id);

      // 5. Delete from memories (CASCADE auto-deletes edges)
      this.db.prepare(`DELETE FROM memories WHERE id = ?`).run(id);

      // 6. Scrub from pending_clusters (all clusters, not just unreviewed,
      //    to avoid ghost IDs in reviewed clusters used for analytics/debugging)
      const clusters = this.db
        .prepare(`SELECT id, memory_ids FROM pending_clusters`)
        .all() as Array<{ id: string; memory_ids: string }>;

      let clustersAffected = 0;
      for (const cluster of clusters) {
        let ids: string[];
        try {
          ids = JSON.parse(cluster.memory_ids);
        } catch {
          continue;
        }
        if (!Array.isArray(ids) || !ids.includes(id)) continue;

        const remaining = ids.filter((mid) => mid !== id);
        if (remaining.length < 2) {
          // Cluster too small to be useful — remove it
          this.db.prepare(`DELETE FROM pending_clusters WHERE id = ?`).run(cluster.id);
        } else {
          this.db
            .prepare(`UPDATE pending_clusters SET memory_ids = ? WHERE id = ?`)
            .run(JSON.stringify(remaining), cluster.id);
        }
        clustersAffected++;
      }

      return { deleted: true, edgesRemoved, supersededRestored, clustersAffected };
    });

    return txn();
  }

  getEmbedding(memoryId: string): number[] | null {
    const row = this.db
      .prepare(`SELECT embedding FROM memory_vectors WHERE memory_id = ?`)
      .get(memoryId) as any;
    if (!row?.embedding) return null;
    return Array.from(new Float32Array(row.embedding.buffer, row.embedding.byteOffset, row.embedding.byteLength / 4));
  }

  close(): void {
    this.db.close();
  }
}
