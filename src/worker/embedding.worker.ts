import type { EmbeddingRequest, EmbeddingResponse } from "./types.js";

/**
 * Message handler for incoming EmbeddingRequests.
 *
 * Exported so the handler can be unit-tested without module-level mocking.
 * Until the ONNX model is wired up (Phase 5), returns a dummy 384-dim vector.
 */
export function handleMessage(
  event: MessageEvent<EmbeddingRequest>,
): void {
  const { id } = event.data;
  const vector = new Float32Array(384).fill(0.1);
  const response: EmbeddingResponse = { id, vector };
  self.postMessage(response);
}

self.addEventListener("message", handleMessage as EventListener);
