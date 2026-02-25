/**
 * Stub module — replaced at build time by the `inline-worker` esbuild plugin
 * in tsup.config.ts with the full content of dist/embedding.worker.bundle.js
 * as a string constant.
 *
 * At runtime the string is used to create a Blob URL so the Worker can be
 * instantiated without any file-path resolution (fixing the Vite node_modules
 * path-resolution bug).
 *
 * In test environments (Vitest/Node) this stub is imported directly; tests that
 * exercise WorkerManager pass an explicit workerUrl, so this value is never
 * used, but the module must exist to satisfy the TypeScript resolver.
 */
export const workerCode = "";
