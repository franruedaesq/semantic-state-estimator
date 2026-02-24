/**
 * Message contract between the Main Thread and the Worker Thread.
 * Every message carries an `id` so responses can be mapped back to their originating Promises.
 */

/** A request sent from the main thread to the embedding worker. */
export interface EmbeddingRequest {
  /** UUID that uniquely identifies this request. */
  id: string;
  /** The raw text to embed. */
  text: string;
}

/** A response sent from the embedding worker back to the main thread. */
export interface EmbeddingResponse {
  /** UUID matching the originating {@link EmbeddingRequest}. */
  id: string;
  /** The computed embedding vector, or null when an error occurred. */
  vector: Float32Array | null;
  /** Human-readable error message, present only when the worker failed. */
  error?: string;
}

/** Status events broadcast by the worker so the UI can reflect model state. */
export type WorkerStatus = "loading" | "ready" | "failed";

export interface WorkerStatusEvent {
  status: WorkerStatus;
}
