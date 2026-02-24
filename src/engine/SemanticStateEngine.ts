import { emaFusion, cosineSimilarity } from "../vectorMath.js";

/**
 * Age-based health decay rate: health lost per millisecond of inactivity.
 * At this rate, age alone reduces health to 0 after ~10 seconds of inactivity.
 */
const AGE_DECAY_RATE = 0.0001;

/**
 * Weight applied to the most-recent drift value when computing healthScore.
 * A drift of 1.0 (orthogonal vectors) reduces health by 0.5.
 */
const DRIFT_WEIGHT = 0.5;

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

/**
 * SemanticStateEngine tracks the implicit semantic intent of an event stream
 * using Exponential Moving Average (EMA) vector fusion.
 *
 * It fires an optional drift callback when incoming embeddings diverge
 * significantly from the current state, and exposes a healthScore that
 * degrades with both age and volatility.
 */
export class SemanticStateEngine {
  private readonly alpha: number;
  private readonly driftThreshold: number;
  private readonly onDriftDetected?: (
    vector: number[],
    driftScore: number,
  ) => void;

  private stateVector: number[];
  private lastUpdatedAt: number;
  private lastDrift: number;
  private updateCount: number;

  constructor(config: SemanticStateEngineConfig) {
    this.alpha = config.alpha;
    this.driftThreshold = config.driftThreshold;
    this.onDriftDetected = config.onDriftDetected;

    this.stateVector = [];
    this.lastUpdatedAt = Date.now();
    this.lastDrift = 0;
    this.updateCount = 0;
  }

  /**
   * Fuses a new embedding into the rolling semantic state using EMA.
   *
   * On the first call the embedding establishes the baseline.
   * On subsequent calls, if the cosine similarity between the current state
   * and the new embedding falls below {@link SemanticStateEngineConfig.driftThreshold},
   * the {@link SemanticStateEngineConfig.onDriftDetected} callback is fired
   * *before* the EMA fusion is applied.
   *
   * @param embedding The new embedding vector E_t.
   */
  update(embedding: number[]): void {
    if (this.updateCount === 0) {
      // First call: establish baseline from a zero-vector origin.
      const zero = new Array(embedding.length).fill(0) as number[];
      this.stateVector = emaFusion(embedding, zero, this.alpha);
      this.lastDrift = 0;
    } else {
      if (embedding.length !== this.stateVector.length) {
        throw new Error(
          `Embedding dimension mismatch: expected ${this.stateVector.length}, got ${embedding.length}`,
        );
      }

      const similarity = cosineSimilarity(this.stateVector, embedding);
      const drift = 1 - similarity;

      if (similarity < this.driftThreshold) {
        this.onDriftDetected?.(embedding, drift);
      }

      this.stateVector = emaFusion(embedding, this.stateVector, this.alpha);
      this.lastDrift = drift;
    }

    this.lastUpdatedAt = Date.now();
    this.updateCount++;
  }

  /**
   * Returns a point-in-time snapshot of the current semantic state.
   */
  getSnapshot(): Snapshot {
    const healthScore = this.calculateHealth();
    return {
      vector: [...this.stateVector],
      healthScore,
      timestamp: this.lastUpdatedAt,
      semanticSummary: this.buildSummary(healthScore),
    };
  }

  /**
   * Computes the current healthScore.
   *
   * Starts at 1.0 and subtracts:
   * - An age penalty proportional to milliseconds elapsed since the last update.
   * - A drift penalty proportional to the most recent drift magnitude.
   *
   * The result is clamped to [0, 1].
   */
  private calculateHealth(): number {
    const timeSinceUpdate = Date.now() - this.lastUpdatedAt;
    const agePenalty = timeSinceUpdate * AGE_DECAY_RATE;
    const driftPenalty = this.lastDrift * DRIFT_WEIGHT;
    return Math.max(0, Math.min(1, 1.0 - agePenalty - driftPenalty));
  }

  private buildSummary(healthScore: number): string {
    if (healthScore > 0.8) return "stable";
    if (healthScore > 0.5) return "drifting";
    return "volatile";
  }
}
