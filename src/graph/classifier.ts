import type { EdgeRelation, Memory } from "../types/index.js";

interface ClassifyResult {
  relation: EdgeRelation;
  strength: number;
  contradicts: boolean;
  reasoning: string;
}

interface ContradictionResult {
  isContradiction: boolean;
  supersededId?: string;
  reasoning: string;
}

/**
 * Uses Claude Haiku to classify the relationship between two memories.
 * Falls back to "relates_to" if no API key or on any error.
 */
export async function classifyEdge(
  newMemory: string,
  existingMemory: string,
  apiKey: string
): Promise<ClassifyResult> {
  const prompt = `You are a memory graph classifier. Classify the relationship between two memories.

Memory A (new): "${newMemory}"
Memory B (existing): "${existingMemory}"

Respond with a JSON object only, no explanation:
{
  "relation": one of ["relates_to", "caused_by", "evolved_from", "led_to", "contradicts"],
  "strength": number from 0.1 to 1.0,
  "contradicts": boolean (true only if these directly conflict or contradict),
  "reasoning": one sentence
}

Rules:
- "contradicts": A directly conflicts with B (e.g., "likes cats" vs "hates cats")
- "evolved_from": A is an updated or refined version of B
- "caused_by": A happened because of B
- "led_to": A caused or contributed to B
- "relates_to": A and B are related but no stronger relation applies`;

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "claude-haiku-4-5",
        max_tokens: 200,
        messages: [{ role: "user", content: prompt }],
      }),
    });

    if (!response.ok) {
      return fallback();
    }

    const data = await response.json() as any;
    const text: string = data.content?.[0]?.text ?? "";

    // Extract JSON from response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return fallback();

    const parsed = JSON.parse(jsonMatch[0]);
    return {
      relation: parsed.relation as EdgeRelation,
      strength: typeof parsed.strength === "number" ? Math.min(1, Math.max(0.1, parsed.strength)) : 0.7,
      contradicts: !!parsed.contradicts,
      reasoning: parsed.reasoning ?? "",
    };
  } catch {
    return fallback();
  }
}

function fallback(): ClassifyResult {
  return { relation: "relates_to", strength: 0.5, contradicts: false, reasoning: "fallback" };
}

/**
 * Checks if a new memory directly contradicts any of the candidate memories.
 * Returns the ID of the memory that should be superseded (the old one).
 */
export async function detectContradiction(
  newContent: string,
  candidates: Memory[],
  apiKey: string
): Promise<ContradictionResult> {
  if (candidates.length === 0) return { isContradiction: false, reasoning: "no candidates" };

  const list = candidates
    .map((m, i) => `${i + 1}. [id:${m.id}] "${m.content}"`)
    .join("\n");

  const prompt = `You are checking if a new memory contradicts any existing memories.

New memory: "${newContent}"

Existing memories:
${list}

Does the new memory directly contradict or replace any existing memory? Respond with JSON only:
{
  "isContradiction": boolean,
  "supersededId": "the id of the memory being replaced, or null",
  "reasoning": "one sentence"
}

Only mark isContradiction=true for direct factual conflicts (e.g., preference flips, decision reversals, changed facts). Additions or elaborations are NOT contradictions.`;

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: "claude-haiku-4-5",
        max_tokens: 200,
        messages: [{ role: "user", content: prompt }],
      }),
    });

    if (!response.ok) return { isContradiction: false, reasoning: "api error" };

    const data = await response.json() as any;
    const text: string = data.content?.[0]?.text ?? "";

    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return { isContradiction: false, reasoning: "parse error" };

    const parsed = JSON.parse(jsonMatch[0]);
    return {
      isContradiction: !!parsed.isContradiction,
      supersededId: parsed.supersededId ?? undefined,
      reasoning: parsed.reasoning ?? "",
    };
  } catch {
    return { isContradiction: false, reasoning: "error" };
  }
}
