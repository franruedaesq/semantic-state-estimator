/**
 * Pure vector math utilities for semantic state estimation.
 *
 * Provides vector addition, scalar multiplication, normalization,
 * cosine similarity, and EMA (Exponential Moving Average) fusion.
 */

import { DimensionMismatchError, SemanticStateError } from "../errors.js";

/** Asserts that two vectors have the same length, throwing otherwise. */
function assertSameDimension(a: number[], b: number[]): void {
  if (a.length !== b.length) {
    throw new DimensionMismatchError(a.length, b.length);
  }
}

/**
 * Adds two vectors element-wise.
 *
 * @param a First vector
 * @param b Second vector
 * @returns  Element-wise sum
 */
export function add(a: number[], b: number[]): number[] {
  assertSameDimension(a, b);
  return a.map((val, i) => val + b[i]!);
}

/**
 * Multiplies every element of a vector by a scalar.
 *
 * @param v      Input vector
 * @param scalar Scalar multiplier
 * @returns      Scaled vector
 */
export function scale(v: number[], scalar: number): number[] {
  return v.map((val) => val * scalar);
}

/**
 * Normalizes a vector to unit length (L2 normalization).
 *
 * @param v Input vector
 * @returns  Unit vector, or zero vector if input magnitude is 0
 */
export function normalize(v: number[]): number[] {
  const mag = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
  if (mag === 0) {
    return v.map(() => 0);
  }
  return v.map((val) => val / mag);
}

/**
 * Computes the cosine similarity between two vectors.
 *
 * @param a First vector
 * @param b Second vector
 * @returns  Cosine similarity in [-1, 1], or 0 if either vector has zero magnitude
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  assertSameDimension(a, b);
  const dot = a.reduce((sum, val, i) => sum + val * b[i]!, 0);
  const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  if (magA === 0 || magB === 0) {
    return 0;
  }
  return dot / (magA * magB);
}

/**
 * Computes the Exponential Moving Average (EMA) fusion of two vectors.
 *
 * Formula: S_t = α · E_t + (1 − α) · S_{t-1}
 *
 * @param current  New embedding vector E_t
 * @param previous Previous state vector S_{t-1}
 * @param alpha    Decay factor α ∈ (0, 1]. Higher values weight recent events more.
 * @returns        Updated state vector S_t
 */
export function emaFusion(
  current: number[],
  previous: number[],
  alpha: number,
): number[] {
  assertSameDimension(current, previous);
  if (alpha <= 0 || alpha > 1) {
    throw new SemanticStateError(`Alpha must be in the range (0, 1], got ${alpha}`);
  }
  return current.map((val, i) => alpha * val + (1 - alpha) * previous[i]!);
}
