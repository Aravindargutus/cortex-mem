# Contributing to Cortex

Thanks for your interest in contributing to Cortex!

## Development setup

```bash
git clone https://github.com/Aravindargutus/cortex-mem.git
cd cortex-mem
npm install
npm run build
```

### Commands

```bash
npm run build       # Build with tsup
npm run dev         # Build in watch mode
npm run typecheck   # Type check without emitting
npm run bench       # Build + run full benchmark suite
```

### Running locally with Claude Desktop

```bash
npm run build
node dist/cli.js setup
# Restart Claude Desktop
```

## How to contribute

1. **Fork** the repo and create a branch from `main`.
2. **Make your changes** — keep commits focused and small.
3. **Run the checks** before submitting:
   ```bash
   npm run typecheck
   npm run bench
   ```
4. **Open a pull request** against `main`.

## Code style

- TypeScript, strict mode
- ESM (`"type": "module"`)
- No external linter — keep it consistent with existing code

## Reporting bugs

Use the [bug report template](https://github.com/Aravindargutus/cortex-mem/issues/new?template=bug_report.yml). Include your Node version, OS, and any relevant logs.

## Feature requests

Use the [feature request template](https://github.com/Aravindargutus/cortex-mem/issues/new?template=feature_request.yml).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
