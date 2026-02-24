import type { EmbeddingRequest, EmbeddingResponse, WorkerInitMessage, WorkerStatusEvent } from "./types.js";

type PendingRequest = {
  resolve: (value: Float32Array) => void;
  reject: (reason: Error) => void;
};

/**
 * WorkerManager wraps a browser Worker in a clean async/await API.
 *
 * It maintains a Map of pending Promises keyed by request UUID so that
 * out-of-order worker responses are still dispatched to the correct caller.
 *
 * Calls to `getEmbedding()` while the worker is not yet ready are dropped,
 * which is acceptable for probabilistic semantic state and avoids memory leaks.
 */
export class WorkerManager {
  private readonly worker: Worker;
  private readonly pendingRequests = new Map<string, PendingRequest>();
  private isReady: boolean = false;

  constructor(
    workerUrl: string | URL = new URL("./embedding.worker.js", import.meta.url),
    modelName: string = "Xenova/all-MiniLM-L6-v2",
  ) {
    this.worker = new Worker(workerUrl, { type: "module" });
    this.worker.onmessage = (event: MessageEvent<EmbeddingResponse | WorkerStatusEvent>) => {
      const data = event.data;
      if ('type' in data) {
        if (data.type === 'STATUS') {
          this.isReady = data.status === 'ready';
          return;
        }
        if (data.type === 'PROGRESS') {
          return;
        }
      }
      const { id, vector, error } = data as EmbeddingResponse;
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

    const initMessage: WorkerInitMessage = { type: "INIT", modelName };
    this.worker.postMessage(initMessage);
  }

  /**
   * Sends `text` to the worker and returns a Promise that resolves with the
   * resulting embedding vector, or rejects if the worker reports an error.
   *
   * If the worker is not yet ready (model still loading), the request is dropped
   * and the Promise rejects immediately to prevent unbounded memory growth.
   *
   * NOTE: A production implementation could instead queue requests and flush them
   * once the worker signals `ready`.
   */
  getEmbedding(text: string): Promise<Float32Array> {
    if (!this.isReady) {
      return Promise.reject(new Error("Worker is not ready"));
    }
    return new Promise<Float32Array>((resolve, reject) => {
      const id = crypto.randomUUID();
      this.pendingRequests.set(id, { resolve, reject });
      const request: EmbeddingRequest = { type: "EMBED", id, text };
      this.worker.postMessage(request);
    });
  }
}
