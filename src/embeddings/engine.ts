import { pipeline, env } from "@xenova/transformers";
import { join } from "path";
import { homedir } from "os";

// Store models in ~/.cortex/models/
env.cacheDir = join(homedir(), ".cortex", "models");

let embedder: any = null;

async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
      quantized: true,
    });
  }
  return embedder;
}

export async function embedLocal(text: string): Promise<number[]> {
  const model = await getEmbedder();
  const output = await model(text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}

export async function embedApi(text: string, apiKey: string, model: string = "text-embedding-3-small"): Promise<number[]> {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ model, input: text }),
  });

  if (!res.ok) {
    throw new Error(`Embedding API error: ${res.status} ${await res.text()}`);
  }

  const data = (await res.json()) as { data?: Array<{ embedding: number[] }> };
  if (!data.data?.[0]?.embedding) {
    throw new Error(`Embedding API returned unexpected response shape`);
  }
  return data.data[0].embedding;
}

export async function embed(
  text: string,
  mode: "local" | "api" = "local",
  apiKey?: string,
  model?: string
): Promise<number[]> {
  if (mode === "api" && apiKey) {
    return embedApi(text, apiKey, model);
  }
  return embedLocal(text);
}
