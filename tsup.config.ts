import { defineConfig } from "tsup";
import fs from "node:fs";
import path from "node:path";
import type { Plugin } from "esbuild";

/**
 * esbuild plugin that replaces the stub `workerCode.ts` module with a module
 * that exports the pre-built worker bundle as a plain string constant.
 *
 * Requires dist/embedding.worker.bundle.js to already exist (produced by the
 * Phase-1 build: `tsup --config tsup.worker.config.ts`).
 */
function inlineWorkerPlugin(): Plugin {
  return {
    name: "inline-worker",
    setup(build) {
      build.onLoad({ filter: /workerCode\.[jt]s$/ }, () => {
        const bundlePath = path.resolve("dist/embedding.worker.bundle.js");
        const code = fs.readFileSync(bundlePath, "utf8");
        // Escape backticks and template-literal markers.
        const escaped = code
          .replace(/\\/g, "\\\\")
          .replace(/`/g, "\\`")
          .replace(/\$\{/g, "\\${");
        return {
          contents: `export const workerCode = \`${escaped}\`;`,
          loader: "js",
        };
      });
    },
  };
}

/**
 * Phase 2 — main library.
 *
 * The esbuild plugin reads dist/embedding.worker.bundle.js (phase 1 output)
 * and inlines it as a string constant wherever `workerCode.ts` is imported.
 * This makes WorkerManager fully portable — no file-URL resolution needed.
 */
export default defineConfig({
  entry: {
    index: "src/index.ts",
    zustand: "src/adapters/zustand.ts",
    react: "src/react/index.ts",
  },
  format: ["esm", "cjs"],
  dts: true,
  splitting: false,
  sourcemap: true,
  external: ["@huggingface/transformers", "zustand", "react"],
  esbuildPlugins: [inlineWorkerPlugin()],
});
