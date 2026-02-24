/**
 * Vector math utilities for semantic state estimation.
 * Pure functions for EMA fusion, cosine similarity, and vector normalization.
 */

/**
 * Computes the Exponential Moving Average (EMA) of two vectors.
 *
 * Formula: S_t = α * E_t + (1 - α) * S_{t-1}
 *
 * @param current  The new embedding vector E_t
 * @param previous The previous state vector S_{t-1}
 * @param alpha    Decay factor α ∈ (0, 1]. Higher values weight recent events more.
 * @returns        The updated state vector S_t
 */
export function emaFusion(
  current: number[],
  previous: number[],
  alpha: number,
): number[] {
  if (current.length !== previous.length) {
    throw new Error(
      `Vector dimension mismatch: current=${current.length}, previous=${previous.length}`,
    );
  }
  if (alpha <= 0 || alpha > 1) {
    throw new Error(`Alpha must be in the range (0, 1], got ${alpha}`);
  }
  return current.map((val, i) => alpha * val + (1 - alpha) * previous[i]!);
}

/**
 * Computes the cosine similarity between two vectors.
 *
 * @param a First vector
 * @param b Second vector
 * @returns  Cosine similarity in [-1, 1], or 0 if either vector has zero magnitude
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(
      `Vector dimension mismatch: a=${a.length}, b=${b.length}`,
    );
  }
  const dot = a.reduce((sum, val, i) => sum + val * b[i]!, 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  if (magA === 0 || magB === 0) {
    return 0;
  }
  return dot / (magA * magB);
}

/**
 * Normalizes a vector to unit length (L2 normalization).
 *
 * @param v The input vector
 * @returns  The normalized vector, or a zero vector if input magnitude is 0
 */
export function normalize(v: number[]): number[] {
  const mag = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
  if (mag === 0) {
    return v.map(() => 0);
  }
  return v.map((val) => val / mag);
}
