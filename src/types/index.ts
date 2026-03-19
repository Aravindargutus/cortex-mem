export type MemoryCategory = "fact" | "preference" | "decision" | "event" | "insight";
export type MemorySource = "explicit" | "consolidated";
export type EdgeRelation = "relates_to" | "caused_by" | "contradicts" | "evolved_from" | "led_to";

export interface Memory {
  id: string;
  content: string;
  category: MemoryCategory;
  source: MemorySource;
  confidence: number;
  created_at: string;
  updated_at: string;
  superseded: boolean;
}

export interface Edge {
  id: string;
  source_id: string;
  target_id: string;
  relation: EdgeRelation;
  strength: number;
  created_at: string;
}

export interface PendingCluster {
  id: string;
  memory_ids: string[];   // parsed from JSON
  created_at: string;
  reviewed: boolean;
}

export interface CortexConfig {
  dbPath: string;
  embeddingMode: "local" | "api";
  apiKey?: string;
  apiEmbeddingModel?: string;
  /** How often the background scheduler clusters memories (ms). Default: 1800000 (30 min) */
  schedulerIntervalMs?: number;
}

export interface RecallResult {
  memories: Array<{
    content: string;
    category: MemoryCategory;
    relevance: number;
    connectedTo?: string[];
  }>;
  summary: string;
  tokenEstimate: number;
}
