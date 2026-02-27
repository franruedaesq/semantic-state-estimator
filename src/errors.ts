/**
 * Custom error classes for semantic-state-estimator.
 *
 * All errors thrown by this library are instances of `SemanticStateError`
 * (or one of its subclasses) so consumers can distinguish them from
 * unrelated runtime errors with a single `instanceof` check.
 */

/**
 * Base error class for all errors emitted by semantic-state-estimator.
 *
 * @example
 * ```ts
 * try {
 *   await engine.update(text);
 * } catch (err) {
 *   if (err instanceof SemanticStateError) {
 *     // library-specific error handling
 *   }
 * }
 * ```
 */
export class SemanticStateError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SemanticStateError";
    // Restore prototype chain in environments that transpile ES6 classes.
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown when two vectors with incompatible dimensions are passed to a
 * vector math function (e.g. `add`, `cosineSimilarity`, `emaFusion`).
 *
 * @example
 * ```ts
 * import { cosineSimilarity, DimensionMismatchError } from "semantic-state-estimator";
 *
 * try {
 *   cosineSimilarity([1, 0], [1, 0, 0]);
 * } catch (err) {
 *   if (err instanceof DimensionMismatchError) {
 *     console.error(err.message); // Vector dimension mismatch: a=2, b=3
 *   }
 * }
 * ```
 */
export class DimensionMismatchError extends SemanticStateError {
  /** Length of the first vector. */
  readonly dimA: number;
  /** Length of the second vector. */
  readonly dimB: number;

  constructor(dimA: number, dimB: number) {
    super(`Vector dimension mismatch: a=${dimA}, b=${dimB}`);
    this.name = "DimensionMismatchError";
    this.dimA = dimA;
    this.dimB = dimB;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown (or passed to `onError`) when the embedding provider fails to
 * return a valid vector for a given input.
 *
 * @example
 * ```ts
 * const middleware = semanticMiddleware(engine, mapper, config, (err) => {
 *   if (err instanceof EmbeddingProviderError) {
 *     metrics.increment("embedding_failures");
 *   }
 * });
 * ```
 */
export class EmbeddingProviderError extends SemanticStateError {
  /** The raw cause that the provider surfaced, if available. */
  readonly cause?: unknown;

  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = "EmbeddingProviderError";
    this.cause = cause;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}
