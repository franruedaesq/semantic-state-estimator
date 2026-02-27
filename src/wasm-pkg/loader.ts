/**
 * Synchronous WASM loader for `semantic-state-core`.
 *
 * Decodes the inlined base64 WASM bytes and calls `initSync` once so that
 * `WasmStateEngine` is ready to use synchronously on first import.
 *
 * The base64 inlining avoids MIME-type issues and bundler path-resolution
 * problems that arise when loading `.wasm` files via URL/fetch.
 */
import { WASM_BASE64 } from "./wasm_inline.js";
import { initSync, WasmStateEngine } from "./semantic_state_core.js";

/** Decodes a base64 string to a Uint8Array, using the fastest available API. */
function base64ToUint8Array(b64: string): Uint8Array {
  if (typeof Buffer !== "undefined") {
    // Node.js
    return Buffer.from(b64, "base64");
  }
  // Browser
  const binaryString = atob(b64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

// Initialise WASM synchronously the first time this module is imported.
initSync({ module: base64ToUint8Array(WASM_BASE64) });

export { WasmStateEngine };
export type { InitOutput } from "./semantic_state_core.js";
