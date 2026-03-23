import { readFileSync, writeFileSync, renameSync, existsSync, mkdirSync } from "fs";
import { join } from "path";
import { homedir, platform } from "os";

interface McpConfig {
  mcpServers: Record<string, { command: string; args: string[] }>;
}

const CORTEX_ENTRY: { command: string; args: string[] } = {
  command: "npx",
  args: ["-y", "@techiesgult/cortex-mem", "serve"],
};

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

export function getClaudeCodeConfigPath(): string {
  return join(homedir(), ".claude.json");
}

function writeMcpConfig(
  configPath: string,
  label: string,
): { success: boolean; message: string } {
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
      message: `Cortex is already configured in ${label}.`,
    };
  }

  config.mcpServers["cortex"] = CORTEX_ENTRY;

  try {
    const tmp = configPath + ".tmp";
    writeFileSync(tmp, JSON.stringify(config, null, 2), { mode: 0o600 });
    renameSync(tmp, configPath);
    return {
      success: true,
      message: `Added Cortex to ${label} config at ${configPath}`,
    };
  } catch {
    return {
      success: false,
      message: `Failed to write config. Please add manually to ${configPath}`,
    };
  }
}

export function autoConfigureClaude(): { success: boolean; message: string } {
  return writeMcpConfig(getClaudeConfigPath(), "Claude Desktop");
}

export function autoConfigureClaudeCode(): { success: boolean; message: string } {
  return writeMcpConfig(getClaudeCodeConfigPath(), "Claude Code");
}
