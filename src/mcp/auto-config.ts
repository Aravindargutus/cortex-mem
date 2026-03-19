import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { join } from "path";
import { homedir, platform } from "os";

interface McpConfig {
  mcpServers: Record<string, { command: string; args: string[] }>;
}

export function getClaudeConfigPath(): string {
  const home = homedir();
  const os = platform();

  if (os === "darwin") {
    return join(home, "Library", "Application Support", "Claude", "claude_desktop_config.json");
  } else if (os === "win32") {
    return join(home, "AppData", "Roaming", "Claude", "claude_desktop_config.json");
  } else {
    return join(home, ".config", "claude", "claude_desktop_config.json");
  }
}

export function autoConfigureClaude(): { success: boolean; message: string } {
  const configPath = getClaudeConfigPath();
  const configDir = join(configPath, "..");

  if (!existsSync(configDir)) {
    mkdirSync(configDir, { recursive: true });
  }

  let config: McpConfig = { mcpServers: {} };

  if (existsSync(configPath)) {
    try {
      const raw = readFileSync(configPath, "utf-8");
      config = JSON.parse(raw);
      if (!config.mcpServers) {
        config.mcpServers = {};
      }
    } catch {
      return {
        success: false,
        message: `Failed to parse existing config at ${configPath}. Please add Cortex manually.`,
      };
    }
  }

  if (config.mcpServers["cortex"]) {
    return {
      success: true,
      message: "Cortex is already configured in Claude Desktop.",
    };
  }

  config.mcpServers["cortex"] = {
    command: "npx",
    args: ["-y", "cortex-ai", "serve"],
  };

  try {
    writeFileSync(configPath, JSON.stringify(config, null, 2));
    return {
      success: true,
      message: `Added Cortex to Claude Desktop config at ${configPath}`,
    };
  } catch {
    return {
      success: false,
      message: `Failed to write config. Please add manually to ${configPath}`,
    };
  }
}
