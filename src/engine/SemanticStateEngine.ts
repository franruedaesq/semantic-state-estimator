import { WasmStateEngine } from "../wasm-pkg/loader.js";

/**
 * A generic embedding provider contract.
 * Any object that can return an embedding vector for a given text satisfies this interface.
 * This includes `WorkerManager` (on-device WebWorker) as well as custom OpenAI, Ollama,
 * or other remote-inference wrappers.
 */
export interface EmbeddingProvider {
  getEmbedding(text: string): Promise<Float32Array | number[]>;
}

/**
 * Configuration for the SemanticStateEngine.
 */
export interface SemanticStateEngineConfig {
  /** EMA decay factor α ∈ (0, 1]. Higher values weight recent embeddings more. */
  alpha: number;

  /** Minimum cosine similarity below which drift is detected and the callback fires. */
  driftThreshold: number;

  /**
   * Optional callback invoked when the incoming embedding drifts beyond the threshold.
   * Fired *before* the EMA fusion is applied.
   *
   * @param vector    The incoming embedding that triggered the drift.
   * @param driftScore Drift magnitude: 1 − cosine_similarity ∈ [0, 2].
   */
  onDriftDetected?: (vector: number[], driftScore: number) => void;

  /**
   * Optional callback invoked after every successful `update`, once the new
   * EMA state has been fused and listeners have been notified.
   *
   * Use this for observability — logging, metrics, or debugging — without
   * coupling to the React/Zustand subscription model.
   *
   * @param snapshot The updated point-in-time snapshot.
   * @param text     The raw text that triggered the update.
   */
  onStateChange?: (snapshot: Snapshot, text: string) => void;
  /**
   * The embedding provider used to obtain embedding vectors asynchronously.
   * Any object implementing `getEmbedding(text: string): Promise<Float32Array | number[]>`
   * satisfies this interface — including `WorkerManager`, or a custom OpenAI / Ollama wrapper.
   */
  provider: EmbeddingProvider;

  /**
   * The name of the embedding model to use.
   * Must match the modelName passed to the WorkerManager so the worker
   * loads the correct model.
   * @default "Xenova/all-MiniLM-L6-v2"
   */
  modelName?: string;
}

/**
 * A point-in-time snapshot of the current semantic state.
 */
export interface Snapshot {
  /** The current EMA state vector. */
  vector: number[];

  /** Reliability indicator in [0, 1]. Degrades with age and high drift. */
  healthScore: number;

  /** Unix timestamp (ms) of the last state update. */
  timestamp: number;

  /** Human-readable description of the current state quality. */
  semanticSummary: string;
}

/** Shape of the object returned by `WasmStateEngine.update()`. */
interface WasmUpdateResult {
  driftDetected: boolean;
  driftScore: number;
  vector: number[];
}

/** Shape of the object returned by `WasmStateEngine.get_snapshot()`. */
interface WasmSnapshot {
  vector: number[];
  healthScore: number;
  timestamp: number;
  semanticSummary: string;
}

/**
 * SemanticStateEngine tracks the implicit semantic intent of an event stream
 * using Exponential Moving Average (EMA) vector fusion.
 *
 * The computationally intensive vector operations (cosine similarity, EMA
 * fusion, health calculation) are delegated to a Rust/WebAssembly core
 * (`WasmStateEngine`) for better performance and type safety, while the
 * public API surface remains identical.
 *
 * It fires an optional drift callback when incoming embeddings diverge
 * significantly from the current state, and exposes a healthScore that
 * degrades with both age and volatility.
 */
export class SemanticStateEngine {
  private readonly onDriftDetected?: (
    vector: number[],
    driftScore: number,
  ) => void;
  private readonly onStateChange?: (snapshot: Snapshot, text: string) => void;
  private readonly provider: EmbeddingProvider;
  readonly modelName: string;

  private readonly wasmEngine: WasmStateEngine;
  private readonly listeners = new Set<() => void>();

  constructor(config: SemanticStateEngineConfig) {
    this.onDriftDetected = config.onDriftDetected;
    this.onStateChange = config.onStateChange;
    this.provider = config.provider;
    this.modelName = config.modelName ?? "Xenova/all-MiniLM-L6-v2";
    this.wasmEngine = new WasmStateEngine(config.alpha, config.driftThreshold);
  }

  /**
   * Obtains an embedding for `text` from the provider and fuses it into
   * the rolling semantic state using EMA (computed in Rust/WASM).
   *
   * On the first call the embedding establishes the baseline.
   * On subsequent calls, if the cosine similarity between the current state
   * and the new embedding falls below {@link SemanticStateEngineConfig.driftThreshold},
   * the {@link SemanticStateEngineConfig.onDriftDetected} callback is fired
   * *before* the EMA fusion is applied.
   *
   * @param text Raw text whose embedding will be fused into the state.
   */
  async update(text: string): Promise<void> {
    const raw = await this.provider.getEmbedding(text);
    if (raw === null) {
      return;
    }
    const embedding = Float32Array.from(raw);
    const result = this.wasmEngine.update(
      embedding,
      Date.now(),
    ) as WasmUpdateResult;

    if (result.driftDetected) {
      this.onDriftDetected?.(Array.from(result.vector), result.driftScore);
    }

    this.listeners.forEach((l) => l());

    if (this.onStateChange) {
      this.onStateChange(this.getSnapshot(), text);
    }
  }

  /**
   * Subscribes to state changes. Returns an unsubscribe function.
   * The listener is called after every successful `update`.
   */
  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Returns a point-in-time snapshot of the current semantic state.
   */
  getSnapshot(): Snapshot {
    const snap = this.wasmEngine.get_snapshot(Date.now()) as WasmSnapshot;
    return {
      vector: Array.from(snap.vector),
      healthScore: snap.healthScore,
      timestamp: snap.timestamp,
      semanticSummary: snap.semanticSummary,
    };
  }
}
