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
        id          TEXT PRIMARY KEY,
        content     TEXT NOT NULL,
        category    TEXT NOT NULL DEFAULT 'fact',
        source      TEXT NOT NULL DEFAULT 'explicit',
        confidence  REAL DEFAULT 1.0,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL,
        superseded  INTEGER DEFAULT 0
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

    const walk = (ids: string[], depth: number) => {
      if (depth > maxHops || ids.length === 0) return;

      for (const id of ids) {
        if (visited.has(id)) continue;
        visited.add(id);

        const memory = this.db.prepare(`SELECT * FROM memories WHERE id = ? AND superseded = 0`).get(id) as any;
        if (!memory) continue;

        const edges = this.getEdgesFrom(id);
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

        const nextIds = edges
          .map((e) => (e.source_id === id ? e.target_id : e.source_id))
          .filter((nid) => !visited.has(nid));

        walk(nextIds, depth + 1);
      }
    };

    walk(startIds, 0);
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
         ORDER BY created_at ASC
         LIMIT ?`
      )
      .all(limit) as Memory[];
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

  close(): void {
    this.db.close();
  }
}
