import { defineConfig } from "tsup";

const external = [
  "better-sqlite3",
  "sqlite-vec",
  "@xenova/transformers",
];

export default defineConfig([
  {
    entry: { index: "src/index.ts" },
    format: ["esm"],
    target: "node18",
    outDir: "dist",
    clean: true,
    dts: true,
    sourcemap: true,
    splitting: false,
    external,
  },
  {
    entry: { cli: "src/cli.ts" },
    format: ["esm"],
    target: "node18",
    outDir: "dist",
    sourcemap: true,
    splitting: false,
    external,
    banner: {
      js: "#!/usr/bin/env node\n",
    },
  },
  {
    entry: { bench: "benchmarks/bench.ts" },
    format: ["esm"],
    target: "node18",
    outDir: "dist",
    sourcemap: true,
    splitting: false,
    external,
  },
]);
