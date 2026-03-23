import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.recommended,
  {
    languageOptions: {
      parserOptions: {
        projectService: {
          allowDefaultProject: ["benchmarks/*.ts"],
          defaultProject: "tsconfig.eslint.json",
        },
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // Allow unused vars prefixed with _ (common pattern for destructuring)
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      // Allow explicit any in cast positions (e.g., SQLite row results)
      "@typescript-eslint/no-explicit-any": "warn",
      // Allow empty catch blocks (used for fallback patterns)
      "no-empty": ["error", { allowEmptyCatch: true }],
      // Enforce no console.log in src (use process.stderr for MCP servers)
      "no-console": ["warn", { allow: ["error", "warn"] }],
    },
  },
  {
    // Relax rules for CLI (console.log is expected) and benchmarks
    files: ["src/cli.ts", "benchmarks/**"],
    rules: {
      "no-console": "off",
    },
  },
  {
    ignores: ["dist/", "node_modules/", "*.js"],
  }
);
