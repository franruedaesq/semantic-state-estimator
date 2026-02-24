import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    zustand: "src/adapters/zustand.ts",
    react: "src/react/index.ts",
    "embedding.worker": "src/worker/embedding.worker.ts",
  },
  format: ["esm", "cjs"],
  dts: true,
  // Disable code splitting so each entry point produces exactly one predictable
  // output file, matching the subpath exports defined in package.json.
  splitting: false,
  sourcemap: true,
  clean: true,
  external: ["@huggingface/transformers", "zustand", "react"],
});
