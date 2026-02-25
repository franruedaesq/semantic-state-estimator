/**
 * Message contract between the Main Thread and the Worker Thread.
 * Every message carries an `id` so responses can be mapped back to their originating Promises.
 */

/** An initialization message sent from the main thread to configure the worker. */
export interface WorkerInitMessage {
  type: "INIT";
  /** The name of the embedding model the worker should load. */
  modelName: string;
}

/** A request sent from the main thread to the embedding worker. */
export interface EmbeddingRequest {
  type: "EMBED";
  /** UUID that uniquely identifies this request. */
  id: string;
  /** The raw text to embed. */
  text: string;
}

/** Union of all messages the worker can receive from the main thread. */
export type WorkerIncomingMessage = WorkerInitMessage | EmbeddingRequest;

/** A response sent from the embedding worker back to the main thread. */
export interface EmbeddingResponse {
  type: "EMBED_RES";
  /** UUID matching the originating {@link EmbeddingRequest}. */
  id: string;
  /** The computed embedding vector, or null when an error occurred. */
  vector: Float32Array | null;
  /** Human-readable error message, present only when the worker failed. */
  error?: string;
}

/** Status events broadcast by the worker so the UI can reflect model state. */
export type WorkerStatusEvent =
  | { type: 'STATUS'; status: 'loading' | 'ready' | 'failed'; error?: string }
  | { type: 'PROGRESS'; file: string; progress: number };
