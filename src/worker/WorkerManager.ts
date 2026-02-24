import type { EmbeddingRequest, EmbeddingResponse } from "./types.js";

type PendingRequest = {
  resolve: (value: Float32Array) => void;
  reject: (reason: Error) => void;
};

/**
 * WorkerManager wraps a browser Worker in a clean async/await API.
 *
 * It maintains a Map of pending Promises keyed by request UUID so that
 * out-of-order worker responses are still dispatched to the correct caller.
 */
export class WorkerManager {
  private readonly worker: Worker;
  private readonly pendingRequests = new Map<string, PendingRequest>();

  constructor(workerUrl: string | URL) {
    this.worker = new Worker(workerUrl);
    this.worker.onmessage = (event: MessageEvent<EmbeddingResponse>) => {
      const { id, vector, error } = event.data;
      const pending = this.pendingRequests.get(id);
      if (!pending) return;
      this.pendingRequests.delete(id);
      if (error !== undefined) {
        pending.reject(new Error(error));
      } else if (vector !== null) {
        pending.resolve(vector);
      } else {
        pending.reject(new Error("Worker returned null vector without an error message"));
      }
    };
  }

  /**
   * Sends `text` to the worker and returns a Promise that resolves with the
   * resulting embedding vector, or rejects if the worker reports an error.
   *
   * NOTE: If the worker is terminated externally or never replies to a request,
   * the corresponding entry in `pendingRequests` will not be cleaned up.
   * A production implementation should add a per-request timeout to prevent
   * unbounded memory growth.
   */
  getEmbedding(text: string): Promise<Float32Array> {
    return new Promise<Float32Array>((resolve, reject) => {
      const id = crypto.randomUUID();
      this.pendingRequests.set(id, { resolve, reject });
      const request: EmbeddingRequest = { id, text };
      this.worker.postMessage(request);
    });
  }
}
