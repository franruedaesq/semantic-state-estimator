import { emaFusion, cosineSimilarity } from "./vectorMath.js";

/**
 * Configuration options for the SemanticStateEngine.
 */
export interface SemanticStateEngineOptions {
  /**
   * The EMA decay factor α ∈ (0, 1].
   * Higher values weight recent embeddings more heavily.
   * @default 0.3
   */
  alpha?: number;

  /**
   * Minimum cosine similarity change required to trigger a state update.
   * Events producing less change than this threshold are ignored (lazy evaluation).
   * @default 0.05
   */
  changeThreshold?: number;

  /**
   * Dimensionality of the embedding vectors.
   * Must match the output dimension of the embedding model.
   * @default 384
   */
  embeddingDimension?: number;
}

/**
 * A snapshot of the current semantic state.
 */
export interface SemanticStateSnapshot {
  /** The current rolling EMA state vector. */
  stateVector: number[];
  /** ISO timestamp of the last state update. */
  lastUpdatedAt: string;
  /** Number of embeddings that have been fused into the state. */
  updateCount: number;
}

/**
 * SemanticStateEngine is the core orchestrator for the semantic state estimator.
 *
 * It maintains a rolling EMA state vector that fuses incoming embedding vectors
 * and applies lazy evaluation via a cosine similarity change threshold.
 */
export class SemanticStateEngine {
  private readonly alpha: number;
  private readonly changeThreshold: number;
  private readonly embeddingDimension: number;
  private stateVector: number[];
  private lastUpdatedAt: string;
  private updateCount: number;

  constructor(options: SemanticStateEngineOptions = {}) {
    this.alpha = options.alpha ?? 0.3;
    this.changeThreshold = options.changeThreshold ?? 0.05;
    this.embeddingDimension = options.embeddingDimension ?? 384;

    if (this.alpha <= 0 || this.alpha > 1) {
      throw new Error(`alpha must be in (0, 1], got ${this.alpha}`);
    }
    if (this.changeThreshold < 0 || this.changeThreshold > 1) {
      throw new Error(
        `changeThreshold must be in [0, 1], got ${this.changeThreshold}`,
      );
    }
    if (this.embeddingDimension <= 0) {
      throw new Error(
        `embeddingDimension must be positive, got ${this.embeddingDimension}`,
      );
    }

    this.stateVector = new Array(this.embeddingDimension).fill(0) as number[];
    this.lastUpdatedAt = new Date().toISOString();
    this.updateCount = 0;
  }

  /**
   * Fuses a new embedding vector into the rolling semantic state using EMA.
   *
   * Applies lazy evaluation: if the cosine similarity between the current state
   * and the new embedding exceeds (1 - changeThreshold), the update is skipped.
   *
   * @param embedding The new embedding vector to fuse
   * @returns `true` if the state was updated, `false` if the update was skipped
   */
  update(embedding: number[]): boolean {
    if (embedding.length !== this.embeddingDimension) {
      throw new Error(
        `Embedding dimension mismatch: expected ${this.embeddingDimension}, got ${embedding.length}`,
      );
    }

    // Lazy evaluation: skip if the semantic shift is below threshold
    const similarity = cosineSimilarity(this.stateVector, embedding);
    if (this.updateCount > 0 && similarity >= 1 - this.changeThreshold) {
      return false;
    }

    this.stateVector = emaFusion(embedding, this.stateVector, this.alpha);
    this.lastUpdatedAt = new Date().toISOString();
    this.updateCount += 1;
    return true;
  }

  /**
   * Returns a snapshot of the current semantic state.
   */
  getSnapshot(): SemanticStateSnapshot {
    return {
      stateVector: [...this.stateVector],
      lastUpdatedAt: this.lastUpdatedAt,
      updateCount: this.updateCount,
    };
  }

  /**
   * Resets the engine to its initial zero state.
   */
  reset(): void {
    this.stateVector = new Array(this.embeddingDimension).fill(0) as number[];
    this.lastUpdatedAt = new Date().toISOString();
    this.updateCount = 0;
  }
}
