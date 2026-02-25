import { defineConfig } from "tsup";

/**
 * Phase 1 — worker bundle.
 * Bundles embedding.worker.ts with ALL its dependencies so the output is a
 * self-contained file that can be served as a Blob URL with no external imports.
 */
export default defineConfig({
    entry: { "embedding.worker.bundle": "src/worker/embedding.worker.ts" },
    format: ["esm"],
    noExternal: [/.*/],
    splitting: false,
    sourcemap: false,
    clean: false,
    outDir: "dist",
});
