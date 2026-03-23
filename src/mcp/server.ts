import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { CortexStorage } from "../storage/database.js";
import { CortexScheduler } from "../agent/scheduler.js";
import { embed, cosineSim } from "../embeddings/engine.js";
import type { CortexConfig, MemoryCategory, MemorySource, Memory, EdgeRelation } from "../types/index.js";

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export class CortexMcpServer {
  private server:    McpServer;
  private storage:   CortexStorage;
  private config:    CortexConfig;
  private scheduler: CortexScheduler;

  constructor(config: CortexConfig) {
    this.config  = config;
    this.storage = new CortexStorage(config.dbPath);
    this.scheduler = new CortexScheduler(this.storage, (msg) => {
      // Write to stderr so it doesn't pollute the stdio MCP stream
      process.stderr.write(msg + "\n");
    });

    this.server = new McpServer({
      name: "cortex",
      version: "0.1.1",
    });

    this.registerTools();
    this.registerResources();
  }

  private registerTools(): void {
    // memory_save — store a fact; Claude provides relationship context, Cortex handles graph storage
    this.server.tool(
      "memory_save",
      "Save a memory about the user. Call this whenever the user shares a preference, makes a decision, mentions a fact about themselves, or when something important happens. If this memory contradicts or updates an existing one (seen in a previous memory_recall or memory_save response), pass contradicts_id with the old memory's ID. Optionally pass relation to classify the edge type. When saving a consolidation insight from memory_consolidate, set source='consolidated'.",
      {
        content: z.string().max(10_000).describe("The memory to save. Be specific and factual. e.g. 'User prefers TypeScript over JavaScript'"),
        category: z.enum(["fact", "preference", "decision", "event", "insight"]).optional().describe("Type of memory. Use 'preference' for likes/dislikes, 'decision' for choices, 'event' for things that happened, 'insight' for patterns."),
        source: z.enum(["explicit", "consolidated"]).optional().describe("Set to 'consolidated' when saving a pattern discovered via memory_consolidate."),
        contradicts_id: z.string().optional().describe("ID of an existing memory this new one contradicts or replaces. The old memory will be marked superseded."),
        relation: z.enum(["relates_to", "caused_by", "evolved_from", "led_to", "contradicts"]).optional().describe("How this memory relates to the contradicted one, or to similar memories in general."),
      },
      async ({ content, category, source, contradicts_id, relation }) => {
        const cat = (category || "fact") as MemoryCategory;
        const src = (source || "explicit") as MemorySource;

        // Generate embedding first for dedup check
        const embedding = await embed(content, this.config.embeddingMode, this.config.apiKey);

        // Check for near-duplicate memories before saving
        const dupes = this.storage.vectorSearch(embedding, 1);
        if (dupes.length > 0 && dupes[0].distance < 0.08) {
          return {
            content: [{
              type: "text" as const,
              text: `Memory already exists: [id:${dupes[0].memory.id}] "${dupes[0].memory.content}". Skipped duplicate.`,
            }],
          };
        }

        // Save the memory
        const memory = this.storage.saveMemory(content, cat, src);

        // Store embedding
        this.storage.saveEmbedding(memory.id, embedding);

        let connections = 0;
        let contradictionNote = "";

        // Handle explicit contradiction from Claude
        if (contradicts_id) {
          if (!UUID_RE.test(contradicts_id)) {
            return {
              content: [{ type: "text" as const, text: `Invalid contradicts_id format: expected a UUID.` }],
            };
          }
          // Verify the target memory still exists (it may have been deleted
          // between recall and this save call)
          const target = this.storage.getMemoryById(contradicts_id);
          if (target) {
            this.storage.supersedeMemory(contradicts_id);
            this.storage.addEdge(memory.id, contradicts_id, "contradicts", 1.0);
            contradictionNote = " Superseded a conflicting memory.";
          } else {
            contradictionNote = " (contradicts_id not found — target may have been deleted.)";
          }
        }

        // Find semantically related memories and create edges
        const related = this.storage.vectorSearch(embedding, 8);
        const edgeRelation = (relation as EdgeRelation) || "relates_to";
        for (const match of related) {
          if (match.memory.id === memory.id) continue;
          if (match.memory.id === contradicts_id) continue; // already linked above
          if (match.distance < 0.5) {
            if (!this.storage.edgeExists(memory.id, match.memory.id)) {
              this.storage.addEdge(memory.id, match.memory.id, edgeRelation, 1 - match.distance);
              connections++;
            }
          }
        }

        // Return top 3 related memories with IDs so Claude can spot contradictions
        const topRelated = related
          .filter((r) => r.memory.id !== memory.id && r.distance < 0.55)
          .slice(0, 3);
        const relatedNote = topRelated.length > 0
          ? `\n\nRelated existing memories (check for conflicts):\n` +
            topRelated.map((r) => `  [id:${r.memory.id}] [${r.memory.category}] "${r.memory.content}"`).join("\n")
          : "";

        const count = this.storage.getMemoryCount();
        return {
          content: [
            {
              type: "text" as const,
              text: `Remembered: "${content}" [${cat}] [id:${memory.id}]. Connected to ${connections} related memories. Total: ${count}.${contradictionNote}${relatedNote}`,
            },
          ],
        };
      }
    );

    // memory_delete — permanently remove a memory and clean up all references
    this.server.tool(
      "memory_delete",
      "Permanently delete a memory by ID. Use when the user explicitly asks to forget something, or to remove incorrect/outdated information. This will also clean up related graph edges, embeddings, and pending clusters. If the deleted memory had contradicted an older memory, that older memory will be restored (un-superseded) automatically.",
      {
        memory_id: z.string().describe("The UUID of the memory to delete. Get this from memory_recall or memory_timeline."),
        confirm: z.boolean().optional().describe("Set to true to confirm deletion. If omitted or false, returns a preview of what will be affected without deleting."),
      },
      async ({ memory_id, confirm }) => {
        if (!UUID_RE.test(memory_id)) {
          return {
            content: [{ type: "text" as const, text: `Invalid memory_id format: expected a UUID.` }],
          };
        }

        const memory = this.storage.getMemoryById(memory_id);
        if (!memory) {
          return {
            content: [{ type: "text" as const, text: `Memory not found: ${memory_id}. It may have already been deleted.` }],
          };
        }

        // Preview mode — show what will be affected without deleting
        if (!confirm) {
          const edges = this.storage.getEdgesFrom(memory_id);
          const contradicts = edges.filter((e) => e.source_id === memory_id && e.relation === "contradicts");
          const preview = [
            `Memory to delete: [id:${memory.id}] [${memory.category}] "${memory.content}"`,
            `Edges that will be removed: ${edges.length}`,
          ];
          if (contradicts.length > 0) {
            preview.push(`Superseded memories that will be restored: ${contradicts.length}`);
          }
          preview.push(`\nCall memory_delete again with confirm=true to proceed.`);
          return {
            content: [{ type: "text" as const, text: preview.join("\n") }],
          };
        }

        // Confirmed — perform the delete
        const result = this.storage.deleteMemory(memory_id);

        if (!result.deleted) {
          return {
            content: [{ type: "text" as const, text: `Memory not found: ${memory_id}. It may have been deleted between preview and confirm.` }],
          };
        }

        const parts = [`Deleted: "${memory.content}" [${memory.category}]`];
        if (result.edgesRemoved > 0) parts.push(`Removed ${result.edgesRemoved} graph edge(s).`);
        if (result.supersededRestored.length > 0) {
          parts.push(`Restored ${result.supersededRestored.length} previously-superseded memory(ies).`);
        }
        if (result.clustersAffected > 0) parts.push(`Updated ${result.clustersAffected} pending cluster(s).`);
        parts.push(`Total memories: ${this.storage.getMemoryCount()}.`);

        return {
          content: [{ type: "text" as const, text: parts.join(" ") }],
        };
      }
    );

    // memory_recall — retrieve relevant memories using graph-walk + vector search
    this.server.tool(
      "memory_recall",
      "Recall memories relevant to the current context. Uses semantic search and knowledge graph traversal to find the most relevant information. Call this at the start of conversations or when context about the user would be helpful.",
      {
        query: z.string().max(2_000).describe("What to recall. Can be a topic, question, or context description. e.g. 'What does the user think about databases?' or 'User preferences for programming languages'"),
        max_results: z.number().optional().describe("Maximum memories to return (default: 10, max: 20)"),
      },
      async ({ query, max_results }) => {
        const topK = Math.min(max_results || 10, 20);

        // Vector search for semantically similar memories
        const embedding = await embed(query, this.config.embeddingMode, this.config.apiKey);
        const vectorResults = this.storage.vectorSearch(embedding, topK);

        if (vectorResults.length === 0) {
          return {
            content: [
              {
                type: "text" as const,
                text: "No memories found. I don't know anything about this topic yet.",
              },
            ],
          };
        }

        // Graph walk from top vector results to find connected memories
        const startIds = vectorResults.slice(0, 3).map((r) => r.memory.id);
        const graphResults = this.storage.graphWalk(startIds, 2);

        // Merge and dedupe results
        const seen = new Set<string>();
        const allMemories: Array<{ id: string; content: string; category: string; relevance: number; depth: number }> = [];

        for (const vr of vectorResults) {
          if (!seen.has(vr.memory.id)) {
            seen.add(vr.memory.id);
            allMemories.push({
              id: vr.memory.id,
              content: vr.memory.content,
              category: vr.memory.category,
              relevance: 1 - vr.distance,
              depth: 0,
            });
          }
        }

        for (const gr of graphResults) {
          if (!seen.has(gr.memory.id)) {
            seen.add(gr.memory.id);
            allMemories.push({
              id: gr.memory.id,
              content: gr.memory.content,
              category: gr.memory.category,
              relevance: 0.5 / (gr.depth + 1),
              depth: gr.depth,
            });
          }
        }

        // Sort by relevance and cap
        allMemories.sort((a, b) => b.relevance - a.relevance);
        const capped = allMemories.slice(0, topK);

        // Format with IDs so Claude can reference them in memory_save
        const lines = capped.map(
          (m) => `- [id:${m.id}] [${m.category}] ${m.content}`
        );

        const tokenEstimate = Math.ceil(lines.join("\n").length / 4);

        // Nudge consolidation if there are enough unconsolidated memories
        const unconsolidatedCount = this.storage.getUnconsolidatedCount();
        const lastRun = this.storage.getLastConsolidationTime();
        const hourAgo = new Date(Date.now() - 60 * 60 * 1000).toISOString();
        const consolidationDue = unconsolidatedCount >= 10 && (!lastRun || lastRun < hourAgo);
        const consolidationNote = consolidationDue
          ? `\n\n[Cortex] ${unconsolidatedCount} memories haven't been consolidated yet. Consider calling memory_consolidate to discover patterns and compress them into insights.`
          : "";

        return {
          content: [
            {
              type: "text" as const,
              text: `Recalled ${capped.length} memories (~${tokenEstimate} tokens):\n\n${lines.join("\n")}${consolidationNote}`,
            },
          ],
        };
      }
    );

    // memory_timeline — show how a topic has evolved over time
    this.server.tool(
      "memory_timeline",
      "Show how beliefs or facts about a topic have evolved over time. Returns memories related to the topic in chronological order with the edges connecting them (contradicts, evolved_from, etc). Use when the user asks 'how did my thinking on X change?' or 'show me my history with X'.",
      {
        topic: z.string().max(2_000).describe("The topic to trace. e.g. 'database choice', 'TypeScript preference', 'Cortex launch strategy'"),
      },
      async ({ topic }) => {
        const embedding = await embed(topic, this.config.embeddingMode, this.config.apiKey);
        const timeline = this.storage.getTimelineMemories(embedding, 15);

        if (timeline.length === 0) {
          return {
            content: [{ type: "text" as const, text: `No memories found about "${topic}".` }],
          };
        }

        const lines = timeline.map((entry) => {
          const date = entry.memory.created_at.slice(0, 10);
          const status = entry.memory.superseded ? " [superseded]" : "";
          const relationEdges = entry.edges
            .filter((e) => e.relation !== "relates_to")
            .map((e) => {
              const direction = e.source_id === entry.memory.id ? `→` : `←`;
              return `${direction}${e.relation}`;
            });
          const edgeNote = relationEdges.length > 0 ? ` (${relationEdges.join(", ")})` : "";
          return `${date} [${entry.memory.category}]${status}${edgeNote}: ${entry.memory.content}`;
        });

        return {
          content: [
            {
              type: "text" as const,
              text: `Timeline for "${topic}" — ${timeline.length} entries:\n\n${lines.join("\n")}`,
            },
          ],
        };
      }
    );

    // memory_consolidate — return clusters of memories for Claude to synthesize into insights
    this.server.tool(
      "memory_consolidate",
      "Consolidate raw memories into patterns and insights. Returns clusters of related memories grouped by similarity. For each cluster, identify any meaningful pattern and save it as a new memory with category='insight'. Call this periodically or when nudged by memory_recall.",
      {
        limit: z.number().optional().describe("Max memories to consolidate in this run (default: 30)"),
      },
      async ({ limit }) => {
        const cap = Math.min(limit || 30, 50);
        const memories = this.storage.getUnconsolidatedMemories(cap);

        if (memories.length < 3) {
          return {
            content: [
              {
                type: "text" as const,
                text: `Not enough memories to consolidate yet (${memories.length} available, need at least 3). Keep using Cortex and check back later.`,
              },
            ],
          };
        }

        // Group memories by vector similarity into clusters of 3–8
        // Use a simple greedy approach: pick an unclustered memory, find its neighbors
        const embeddings = new Map<string, number[]>();
        for (const m of memories) {
          const emb = await embed(m.content, this.config.embeddingMode, this.config.apiKey);
          embeddings.set(m.id, emb);
        }

        const clustered = new Set<string>();
        const clusters: Memory[][] = [];

        for (const m of memories) {
          if (clustered.has(m.id)) continue;
          const anchor = embeddings.get(m.id)!;
          const cluster: Memory[] = [m];
          clustered.add(m.id);

          for (const other of memories) {
            if (clustered.has(other.id) || cluster.length >= 8) continue;
            const otherEmb = embeddings.get(other.id)!;
            const sim = cosineSim(anchor, otherEmb);
            if (sim > 0.7) {
              cluster.push(other);
              clustered.add(other.id);
            }
          }

          if (cluster.length >= 2) clusters.push(cluster);
        }

        if (clusters.length === 0) {
          return {
            content: [
              {
                type: "text" as const,
                text: "Memories don't cluster into clear groups yet. They may be too diverse or too few.",
              },
            ],
          };
        }

        this.storage.setLastConsolidationTime(new Date().toISOString());

        // Mark all clustered memories as consolidated
        const allClusteredIds = clusters.flatMap((c) => c.map((m) => m.id));
        this.storage.markMemoriesConsolidated(allClusteredIds);

        const clusterText = clusters.map((cluster, i) => {
          const items = cluster.map((m) => `  - [${m.category}] ${m.content}`).join("\n");
          return `Cluster ${i + 1} (${cluster.length} memories):\n${items}`;
        }).join("\n\n");

        return {
          content: [
            {
              type: "text" as const,
              text: `Found ${clusters.length} memory clusters from ${memories.length} memories.\n\nFor each cluster below, identify any meaningful pattern. If you find one, save it with memory_save(category='insight', source implicitly becomes 'consolidated').\n\n${clusterText}\n\nAfter saving insights, reply to the user with a summary of what patterns you found — this is the "while I was thinking" moment.`,
            },
          ],
        };
      }
    );

    // memory_proactive — surface clusters pre-computed by the background scheduler
    this.server.tool(
      "memory_proactive",
      "Check for memory patterns that the background scheduler has already discovered while you were idle. Returns pre-clustered groups of related memories for you to synthesise into insights. Call this at conversation start or periodically. After synthesising, call memory_save with category='insight' for each meaningful pattern.",
      {},
      async () => {
        const pending = this.storage.getPendingClusterCount();
        if (pending === 0) {
          const count = this.storage.getMemoryCount();
          return {
            content: [{ type: "text" as const, text: `No pending patterns. Cortex has ${count} memories and nothing new to surface right now. The background scheduler runs every ${Math.round((this.config.schedulerIntervalMs ?? 1_800_000) / 60_000)} minutes.` }],
          };
        }

        const unreviewed = this.storage.getUnreviewedClusters(5);

        // Mark all returned clusters as reviewed — Claude will process them in this turn
        for (const { cluster } of unreviewed) {
          this.storage.markClusterReviewed(cluster.id);
        }

        const clusterText = unreviewed.map(({ cluster, memories }, i) => {
          const items = memories.map((m) => `  - [id:${m.id}] [${m.category}] ${m.content}`).join("\n");
          return `Pattern opportunity ${i + 1} (${memories.length} related memories, discovered ${cluster.created_at.slice(0, 10)}):\n${items}`;
        }).join("\n\n");

        const remaining = pending - unreviewed.length;
        const remainingNote = remaining > 0 ? `\n\n(${remaining} more cluster(s) pending — call memory_proactive again after saving these.)` : "";

        return {
          content: [
            {
              type: "text" as const,
              text: `The background scheduler found ${unreviewed.length} memory pattern(s) while idle.\n\nFor each group below, identify any meaningful insight and save it with memory_save(category='insight', source='consolidated').\n\n${clusterText}${remainingNote}`,
            },
          ],
        };
      }
    );
  }

  private registerResources(): void {
    this.server.resource(
      "memory-stats",
      "cortex://stats",
      async (uri) => ({
        contents: [
          {
            uri: uri.href,
            mimeType: "text/plain",
            text: `Cortex Memory Stats:\n- Total memories: ${this.storage.getMemoryCount()}\n- Embedding mode: ${this.config.embeddingMode}`,
          },
        ],
      })
    );
  }

  async start(): Promise<void> {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    // Start background scheduler after MCP handshake completes
    this.scheduler.start(this.config.schedulerIntervalMs ?? 30 * 60 * 1000);
  }

  close(): void {
    this.scheduler.stop();
    this.storage.close();
  }
}
