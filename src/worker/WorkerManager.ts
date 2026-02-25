import type { EmbeddingRequest, EmbeddingResponse, WorkerInitMessage, WorkerStatusEvent } from "./types.js";
import { workerCode } from "./workerCode.js";

type PendingRequest = {
  resolve: (value: Float32Array) => void;
  reject: (reason: Error) => void;
};

/**
 * Creates a Blob URL from the inlined worker bundle string.
 *
 * This avoids the Vite dev-mode bug where `new URL('./embedding.worker.js',
 * import.meta.url)` resolves relative to `.vite/deps/` when the library is
 * consumed from node_modules, producing a 404.
 *
 * The blob is self-contained — all dependencies were bundled in at build time
 * by the Phase-1 tsup config — so it works without any external imports.
 */
function createBlobWorkerUrl(): string {
  const blob = new Blob([workerCode], { type: "text/javascript" });
  return URL.createObjectURL(blob);
}

/**
 * WorkerManager wraps a browser Worker in a clean async/await API.
 *
 * It maintains a Map of pending Promises keyed by request UUID so that
 * out-of-order worker responses are still dispatched to the correct caller.
 *
 * By default the worker is created from an inlined Blob URL, which is fully
 * portable across bundlers and does not rely on `import.meta.url` path
 * resolution. Pass an explicit `workerUrl` to override (e.g. for testing or
 * advanced use cases).
 *
 * Calls to `getEmbedding()` while the worker is not yet ready are silently
 * dropped (returning null) to avoid unhandled promise rejections during the
 * model loading phase, which is acceptable for probabilistic semantic state.
 */
export class WorkerManager {
  private readonly worker: Worker;
  private readonly pendingRequests = new Map<string, PendingRequest>();
  private isReady: boolean = false;

  constructor(
    workerUrl?: string | URL,
    modelName: string = "Xenova/all-MiniLM-L6-v2",
  ) {
    const url = workerUrl ?? createBlobWorkerUrl();
    this.worker = new Worker(url, { type: "classic" });

    this.worker.onerror = (event: ErrorEvent) => {
      console.error(
        "SemanticStateEstimator: worker encountered a fatal error — " +
        "embeddings will be unavailable.",
        event,
      );
      this.isReady = false;
    };

    this.worker.onmessage = (event: MessageEvent<EmbeddingResponse | WorkerStatusEvent>) => {
      const data = event.data;
      switch (data.type) {
        case 'STATUS':
          this.isReady = data.status === 'ready';
          return;
        case 'PROGRESS':
          return;
        case 'EMBED_RES': {
          const { id, vector, error } = data;
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
          return;
        }
      }
    };

    const initMessage: WorkerInitMessage = { type: "INIT", modelName };
    this.worker.postMessage(initMessage);
  }

  /**
   * Sends `text` to the worker and returns a Promise that resolves with the
   * resulting embedding vector, or rejects if the worker reports an error.
   *
   * If the worker is not yet ready (model still loading), the request is
   * silently dropped and the Promise resolves with `null` to avoid
   * unhandled promise rejections during the initial page load.
   */
  getEmbedding(text: string): Promise<Float32Array | null> {
    if (!this.isReady) {
      console.warn("SemanticStateEngine: Worker still loading, dropping early event.");
      return Promise.resolve(null);
    }
    return new Promise<Float32Array>((resolve, reject) => {
      const id = crypto.randomUUID();
      this.pendingRequests.set(id, { resolve, reject });
      const request: EmbeddingRequest = { type: "EMBED", id, text };
      this.worker.postMessage(request);
    });
  }
}
