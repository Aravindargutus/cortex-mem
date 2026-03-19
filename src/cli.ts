import { join } from "path";
import { homedir } from "os";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { CortexMcpServer } from "./mcp/server.js";
import type { CortexConfig } from "./types/index.js";

// Suppress only the ExperimentalWarning for node:sqlite — not all warnings
process.on("warning", (w) => {
  if (w.name === "ExperimentalWarning" && w.message.includes("sqlite")) return;
  process.stderr.write(`${w.name}: ${w.message}\n`);
});

const CORTEX_DIR = join(homedir(), ".cortex");
const CONFIG_PATH = join(CORTEX_DIR, "config.json");
const DB_PATH = join(CORTEX_DIR, "memory.db");

function getDefaultConfig(): CortexConfig {
  return {
    dbPath: DB_PATH,
    embeddingMode: "local",
  };
}

function loadConfig(): CortexConfig {
  if (existsSync(CONFIG_PATH)) {
    const raw = readFileSync(CONFIG_PATH, "utf-8");
    return { ...getDefaultConfig(), ...JSON.parse(raw) };
  }
  return getDefaultConfig();
}

function saveConfig(config: CortexConfig): void {
  if (!existsSync(CORTEX_DIR)) {
    mkdirSync(CORTEX_DIR, { recursive: true });
  }
  writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2));
}

function autoConfigureClaude(): boolean {
  const claudeConfigDir = join(homedir(), "Library", "Application Support", "Claude");
  const claudeConfigPath = join(claudeConfigDir, "claude_desktop_config.json");

  let config: any = {};
  if (existsSync(claudeConfigPath)) {
    try {
      config = JSON.parse(readFileSync(claudeConfigPath, "utf-8"));
    } catch {
      // Malformed config — start fresh rather than crashing
      config = {};
    }
  }

  if (!config.mcpServers) {
    config.mcpServers = {};
  }

  // Check if already configured
  if (config.mcpServers.cortex) {
    return false;
  }

  config.mcpServers.cortex = {
    command: "npx",
    args: ["-y", "cortex-ai", "serve"],
  };

  if (!existsSync(claudeConfigDir)) {
    mkdirSync(claudeConfigDir, { recursive: true });
  }

  writeFileSync(claudeConfigPath, JSON.stringify(config, null, 2));
  return true;
}

async function main() {
  const args = process.argv.slice(2);
  const command = args[0] || "serve";

  // Ensure cortex directory exists
  if (!existsSync(CORTEX_DIR)) {
    mkdirSync(CORTEX_DIR, { recursive: true });
  }

  if (command === "serve") {
    const config = loadConfig();
    saveConfig(config);

    const server = new CortexMcpServer(config);

    process.on("SIGINT", () => {
      server.close();
      process.exit(0);
    });

    process.on("SIGTERM", () => {
      server.close();
      process.exit(0);
    });

    await server.start();
  } else if (command === "setup") {
    const config = loadConfig();
    saveConfig(config);

    console.log("\n  🧠 Cortex v0.1.0\n");
    console.log(`  ✓ Config saved to ${CONFIG_PATH}`);
    console.log(`  ✓ Memory database: ${config.dbPath}`);
    console.log(`  ✓ Embedding mode: ${config.embeddingMode}\n`);

    const configured = autoConfigureClaude();
    if (configured) {
      console.log("  ✓ Added to Claude Desktop MCP config\n");
      console.log("  Restart Claude Desktop to connect Cortex.\n");
    } else {
      console.log("  ✓ Claude Desktop already configured\n");
    }
  } else if (command === "stats") {
    const { CortexStorage } = await import("./storage/database.js");
    const config = loadConfig();
    const storage = new CortexStorage(config.dbPath);
    const count = storage.getMemoryCount();
    const pending = storage.getPendingClusterCount();
    console.log(`\n  \u{1F9E0} Cortex Memory Stats`);
    console.log(`  Memories:         ${count}`);
    console.log(`  Pending patterns: ${pending}`);
    console.log(`  Database:         ${config.dbPath}`);
    console.log(`  Embedding:        ${config.embeddingMode}\n`);
    storage.close();
  } else if (command === "daemon") {
    // Stand-alone background agent — runs scheduler independently of Claude Desktop
    const { CortexStorage } = await import("./storage/database.js");
    const { CortexScheduler } = await import("./agent/scheduler.js");

    const config = loadConfig();
    const onceOnly = args.includes("--once");
    const intervalArg = args.find((a) => a.startsWith("--interval="));
    const intervalMs = intervalArg
      ? parseInt(intervalArg.split("=")[1], 10) * 1000
      : (config.schedulerIntervalMs ?? 30 * 60 * 1000);

    const storage = new CortexStorage(config.dbPath);
    const scheduler = new CortexScheduler(storage, (msg) => console.log(msg));

    console.log(`\n  \u{1F9E0} Cortex Daemon`);
    console.log(`  Database: ${config.dbPath}`);

    if (onceOnly) {
      console.log("  Mode: single tick\n");
      const result = await scheduler.tick();
      console.log(`\n  Done. Scanned ${result.memoriesScanned} memories, saved ${result.clustersSaved} pattern(s).`);
      if (result.skipped) console.log(`  Skipped: ${result.reason}`);
      storage.close();
      return;
    }

    console.log(`  Mode: continuous (every ${Math.round(intervalMs / 60_000)} min)`);
    console.log("  Press Ctrl+C to stop.\n");

    scheduler.start(intervalMs, 1000);

    process.on("SIGINT",  () => { scheduler.stop(); storage.close(); process.exit(0); });
    process.on("SIGTERM", () => { scheduler.stop(); storage.close(); process.exit(0); });

    // Keep process alive
    await new Promise(() => {});
  } else {
    console.log("\n  🧠 Cortex — AI Memory Engine\n");
    console.log("  Commands:");
    console.log("    npx cortex-ai setup              Auto-configure Claude Desktop");
    console.log("    npx cortex-ai serve              Start MCP server (used by Claude)");
    console.log("    npx cortex-ai stats              Show memory statistics");
    console.log("    npx cortex-ai daemon             Run background scheduler continuously");
    console.log("    npx cortex-ai daemon --once      Single scheduler tick and exit");
    console.log("    npx cortex-ai daemon --interval=300  Tick every 5 minutes\n");
  }
}

main().catch((err) => {
  console.error("Cortex error:", err.message);
  process.exit(1);
});
