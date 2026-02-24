import type { EmbeddingRequest, EmbeddingResponse, WorkerIncomingMessage, WorkerInitMessage } from "./types.js";

let modelName: string = "Xenova/all-MiniLM-L6-v2";

/** Returns the model name currently configured in the worker. */
export function getModelName(): string {
  return modelName;
}

/**
 * Handles an INIT message, saving the model name to the worker's local state.
 * This should be called exactly once, before any EMBED messages are processed.
 */
export function handleInitMessage(event: MessageEvent<WorkerInitMessage>): void {
  modelName = event.data.modelName;
}

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

self.addEventListener("message", (event: Event) => {
  const msg = (event as MessageEvent<WorkerIncomingMessage>).data;
  if (msg.type === "INIT") {
    handleInitMessage(event as MessageEvent<WorkerInitMessage>);
  } else {
    handleMessage(event as MessageEvent<EmbeddingRequest>);
  }
});
