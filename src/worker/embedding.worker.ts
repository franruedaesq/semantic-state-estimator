import { pipeline, env } from "@huggingface/transformers";
import type { EmbeddingRequest, EmbeddingResponse, WorkerIncomingMessage, WorkerInitMessage } from "./types.js";

// Disable local models; always load from the HuggingFace Hub.
env.allowLocalModels = false;

export class PipelineSingleton {
  static instance: Promise<any> | null = null;
  static modelName: string | null = null;

  static async getInstance(modelName: string, progressCallback?: (data: any) => void) {
    if (this.instance === null || this.modelName !== modelName) {
      this.modelName = modelName;
      this.instance = pipeline('feature-extraction', modelName, {
        dtype: 'q8',
        progress_callback: progressCallback,
      });
    }
    return this.instance;
  }
}

let currentModelName: string = "Xenova/all-MiniLM-L6-v2";

/** Returns the model name currently configured in the worker. */
export function getModelName(): string {
  return currentModelName;
}

/**
 * Handles an INIT message: saves the model name and starts loading the pipeline,
 * broadcasting STATUS events so the main thread can track the model lifecycle.
 */
export async function handleInitMessage(event: MessageEvent<WorkerInitMessage>): Promise<void> {
  currentModelName = event.data.modelName;
  self.postMessage({ type: 'STATUS', status: 'loading' });
  try {
    await PipelineSingleton.getInstance(currentModelName, (data: any) => {
      if (data.status === 'progress' && data.file && data.progress !== undefined) {
        self.postMessage({ type: 'PROGRESS', file: data.file, progress: data.progress });
      }
    });
    self.postMessage({ type: 'STATUS', status: 'ready' });
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    self.postMessage({ type: 'STATUS', status: 'failed', error });
  }
}

/**
 * Handles an EMBED message: runs the text through the pipeline and posts back
 * the resulting normalized 1D Float32Array.
 */
export async function handleMessage(event: MessageEvent<EmbeddingRequest>): Promise<void> {
  const { id, text } = event.data;
  try {
    const extractor = await PipelineSingleton.getInstance(currentModelName);
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    const response: EmbeddingResponse = { type: "EMBED_RES", id, vector: output.data as Float32Array };
    self.postMessage(response);
  } catch (err) {
    const error = err instanceof Error ? err.message : String(err);
    const response: EmbeddingResponse = { type: "EMBED_RES", id, vector: null, error };
    self.postMessage(response);
  }
}

self.addEventListener("message", (event: Event) => {
  const msg = (event as MessageEvent<WorkerIncomingMessage>).data;
  if (msg.type === "INIT") {
    handleInitMessage(event as MessageEvent<WorkerInitMessage>);
  } else {
    handleMessage(event as MessageEvent<EmbeddingRequest>);
  }
});
