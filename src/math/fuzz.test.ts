/**
 * Fuzz / Edge-Case Tests
 *
 * Verifies that the vector math functions and the WasmStateEngine handle
 * edge-case inputs gracefully, matching the guarantees documented in the
 * problem statement:
 *
 *  - Zero vectors: no division-by-zero in normalization / cosine similarity.
 *  - Orthogonal vectors: cosine similarity = 0.0.
 *  - Collinear vectors: cosine similarity = 1.0 (or −1.0).
 *  - Tiny / huge magnitudes: no overflow or underflow.
 *  - Empty embeddings: WasmStateEngine returns an error, not a crash.
 *  - Dimension mismatch: WasmStateEngine returns an error on the second update.
 */

import { describe, it, expect } from "vitest";
import {
  cosineSimilarity,
  normalize,
  emaFusion,
  add,
  scale,
} from "./vector.js";
import { WasmStateEngine } from "../wasm-pkg/loader.js";

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Returns a Float32Array of `len` elements all set to `value`. */
function fill(len: number, value: number): Float32Array {
  return new Float32Array(len).fill(value);
}

/** Returns a number[] of `len` elements all set to `value`. */
function fillArr(len: number, value: number): number[] {
  return new Array(len).fill(value) as number[];
}

const T0 = 1_000_000;

// ── Vector Math Edge Cases (TypeScript utilities) ─────────────────────────────

describe("Fuzz: cosineSimilarity edge cases", () => {
  it("returns 0 for a zero vector vs any vector (no division-by-zero)", () => {
    expect(cosineSimilarity([0, 0, 0, 0], [1, 2, 3, 4])).toBe(0);
    expect(cosineSimilarity([1, 2, 3, 4], [0, 0, 0, 0])).toBe(0);
    expect(cosineSimilarity([0, 0], [0, 0])).toBe(0);
  });

  it("returns 0 for orthogonal vectors (regardless of dimension)", () => {
    // 2-D
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
    // 4-D
    expect(cosineSimilarity([1, 0, 0, 0], [0, 1, 0, 0])).toBeCloseTo(0);
    expect(cosineSimilarity([0, 0, 1, 0], [0, 1, 0, 0])).toBeCloseTo(0);
    // 8-D
    expect(
      cosineSimilarity(
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
      ),
    ).toBeCloseTo(0);
  });

  it("returns 1 for identical (collinear) vectors", () => {
    expect(cosineSimilarity([3, 4], [3, 4])).toBeCloseTo(1);
    expect(cosineSimilarity([1, 1, 1, 1], [1, 1, 1, 1])).toBeCloseTo(1);
    // Scaled copies are also collinear.
    expect(cosineSimilarity([1, 2, 3], [2, 4, 6])).toBeCloseTo(1);
  });

  it("returns -1 for anti-collinear (opposite direction) vectors", () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1);
    expect(cosineSimilarity([1, 2, 3], [-2, -4, -6])).toBeCloseTo(-1);
  });

  it("handles tiny magnitudes without underflow (returns a valid number)", () => {
    const tiny = 1e-20;
    const a = [tiny, 0, 0, 0];
    const b = [tiny, 0, 0, 0];
    const result = cosineSimilarity(a, b);
    expect(isFinite(result)).toBe(true);
    expect(result).toBeCloseTo(1);
  });

  it("handles huge magnitudes without overflow (returns a valid number)", () => {
    const huge = 1e30;
    const a = [huge, 0, 0];
    const b = [huge, 0, 0];
    const result = cosineSimilarity(a, b);
    expect(isFinite(result)).toBe(true);
    expect(result).toBeCloseTo(1);
  });

  it("handles high-dimensional vectors (384-d) correctly for orthogonal pair", () => {
    const dim = 384;
    const a = new Array(dim).fill(0) as number[];
    const b = new Array(dim).fill(0) as number[];
    a[0] = 1;
    b[1] = 1;
    expect(cosineSimilarity(a, b)).toBeCloseTo(0);
  });

  it("handles high-dimensional vectors (384-d) correctly for identical pair", () => {
    const dim = 384;
    const v = fillArr(dim, 1 / Math.sqrt(dim)); // unit vector
    expect(cosineSimilarity(v, v)).toBeCloseTo(1);
  });
});

describe("Fuzz: normalize edge cases", () => {
  it("returns a zero vector for a zero input without throwing", () => {
    expect(normalize([0, 0, 0, 0])).toEqual([0, 0, 0, 0]);
    expect(normalize([0])).toEqual([0]);
  });

  it("produces a unit vector for a tiny-magnitude input", () => {
    const tiny = 1e-30;
    const result = normalize([tiny, 0, 0]);
    const mag = Math.sqrt(result.reduce((s, x) => s + x * x, 0));
    expect(isFinite(mag)).toBe(true);
    // Direction should be preserved: result[0] ≈ 1
    expect(result[0]).toBeCloseTo(1);
  });

  it("produces a unit vector for a huge-magnitude input (within JS float64 safe range)", () => {
    // 1e150 squared = 1e300, which is still finite (MAX_VALUE ≈ 1.8e308).
    const huge = 1e150;
    const result = normalize([0, huge, 0]);
    const mag = Math.sqrt(result.reduce((s, x) => s + x * x, 0));
    expect(isFinite(mag)).toBe(true);
    expect(result[1]).toBeCloseTo(1);
  });

  it("produces a unit vector for a 384-d uniform input", () => {
    const dim = 384;
    const v = fillArr(dim, 1);
    const result = normalize(v);
    const mag = Math.sqrt(result.reduce((s, x) => s + x * x, 0));
    expect(mag).toBeCloseTo(1);
  });
});

describe("Fuzz: emaFusion edge cases", () => {
  it("with alpha=1 always returns the current embedding exactly", () => {
    const current = [1, 2, 3, 4];
    const previous = [9, 8, 7, 6];
    expect(emaFusion(current, previous, 1)).toEqual(current);
  });

  it("with a near-zero alpha the previous state dominates", () => {
    const current = fillArr(4, 100);
    const previous = fillArr(4, 1);
    const alpha = 0.001;
    const result = emaFusion(current, previous, alpha);
    // 0.001 * 100 + 0.999 * 1 = 1.099
    result.forEach((v) => expect(v).toBeCloseTo(1.099));
  });

  it("handles huge values without producing NaN or Infinity", () => {
    const huge = 1e200;
    const current = fillArr(4, huge);
    const previous = fillArr(4, huge);
    const result = emaFusion(current, previous, 0.5);
    result.forEach((v) => {
      expect(isFinite(v)).toBe(true);
    });
  });

  it("handles tiny values without underflow to zero", () => {
    const tiny = 1e-200;
    const current = fillArr(4, tiny);
    const previous = fillArr(4, tiny);
    const result = emaFusion(current, previous, 0.5);
    result.forEach((v) => {
      expect(v).toBe(tiny); // 0.5*tiny + 0.5*tiny = tiny
    });
  });
});

describe("Fuzz: add and scale edge cases", () => {
  it("add returns zero vector when both inputs are zero", () => {
    expect(add([0, 0, 0], [0, 0, 0])).toEqual([0, 0, 0]);
  });

  it("scale by 0 returns zero vector", () => {
    expect(scale([1, 2, 3, 4], 0)).toEqual([0, 0, 0, 0]);
  });

  it("scale handles large dimension vectors", () => {
    const dim = 384;
    const v = fillArr(dim, 1);
    const result = scale(v, 2);
    result.forEach((x) => expect(x).toBe(2));
  });
});

// ── WasmStateEngine Edge Cases ────────────────────────────────────────────────

describe("Fuzz: WasmStateEngine edge cases", () => {
  it("throws (or returns error JsValue) for an empty Float32Array", () => {
    const engine = new WasmStateEngine(0.5, 0.75);
    // Empty array has length 0 — WasmStateEngine should reject it.
    expect(() => engine.update(new Float32Array(0), T0)).toThrow();
  });

  it("throws (or returns error JsValue) on dimension mismatch after first update", () => {
    const engine = new WasmStateEngine(0.5, 0.75);
    engine.update(new Float32Array([1, 0, 0, 0]), T0);
    // Different dimension on second call.
    expect(() => engine.update(new Float32Array([1, 0]), T0 + 1)).toThrow();
  });

  it("handles a zero vector as first embedding without crashing", () => {
    const engine = new WasmStateEngine(0.5, 0.75);
    expect(() => engine.update(new Float32Array([0, 0, 0, 0]), T0)).not.toThrow();
    const snap = engine.get_snapshot(T0) as { vector: number[]; healthScore: number };
    expect(snap.vector).toHaveLength(4);
    expect(isFinite(snap.healthScore)).toBe(true);
  });

  it("handles a zero vector as second embedding without crashing (zero cosine similarity)", () => {
    const engine = new WasmStateEngine(0.5, 0.75);
    engine.update(new Float32Array([1, 0, 0, 0]), T0);
    expect(() => engine.update(new Float32Array([0, 0, 0, 0]), T0 + 1)).not.toThrow();
    const snap = engine.get_snapshot(T0 + 1) as { vector: number[]; healthScore: number };
    expect(isFinite(snap.healthScore)).toBe(true);
  });

  it("handles tiny-magnitude embeddings without producing NaN health scores", () => {
    const engine = new WasmStateEngine(0.5, 0.75);
    const tiny = fill(4, 1e-20);
    engine.update(tiny, T0);
    const snap = engine.get_snapshot(T0) as { healthScore: number };
    expect(isFinite(snap.healthScore)).toBe(true);
  });

  it("handles large embeddings without producing NaN or Infinity health scores", () => {
    const engine = new WasmStateEngine(0.5, 0.75);
    // f32 max is ~3.4e38; use 1e20 to stay safe.
    const large = fill(4, 1e20);
    engine.update(large, T0);
    const snap = engine.get_snapshot(T0) as { healthScore: number };
    expect(isFinite(snap.healthScore)).toBe(true);
  });

  it("handles orthogonal embeddings and reports driftScore close to 1.0", () => {
    const engine = new WasmStateEngine(0.5, 0.75);
    engine.update(new Float32Array([1, 0, 0, 0]), T0);
    const result = engine.update(
      new Float32Array([0, 1, 0, 0]),
      T0 + 1000,
    ) as { driftDetected: boolean; driftScore: number };
    expect(result.driftDetected).toBe(true);
    expect(result.driftScore).toBeCloseTo(1.0, 4);
  });

  it("handles collinear embeddings and reports driftScore close to 0.0", () => {
    const engine = new WasmStateEngine(0.5, 0.75);
    engine.update(new Float32Array([1, 0, 0, 0]), T0);
    const result = engine.update(
      new Float32Array([2, 0, 0, 0]), // same direction, different magnitude
      T0 + 1000,
    ) as { driftDetected: boolean; driftScore: number };
    // cosine similarity ≈ 1 → driftScore ≈ 0
    expect(result.driftScore).toBeCloseTo(0.0, 4);
    expect(result.driftDetected).toBe(false);
  });

  it("healthScore is always clamped to [0, 1] after many updates", () => {
    const engine = new WasmStateEngine(0.5, 0.75);
    // Feed highly volatile embeddings to maximise drift penalty.
    const embeddings = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [-1, 0, 0, 0],
      [0, -1, 0, 0],
    ];
    for (let i = 0; i < embeddings.length; i++) {
      const now = T0 + i * 1000;
      engine.update(new Float32Array(embeddings[i]!), now);
      const snap = engine.get_snapshot(now) as { healthScore: number };
      expect(snap.healthScore).toBeGreaterThanOrEqual(0);
      expect(snap.healthScore).toBeLessThanOrEqual(1);
    }
  });

  it("semanticSummary is one of the three valid strings", () => {
    const validSummaries = new Set(["stable", "drifting", "volatile"]);
    const engine = new WasmStateEngine(0.5, 0.75);
    const embeddings = [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [1, 0, 0, 0],
    ];
    for (let i = 0; i < embeddings.length; i++) {
      engine.update(new Float32Array(embeddings[i]!), T0 + i * 100_000);
      const snap = engine.get_snapshot(T0 + i * 100_000) as { semanticSummary: string };
      expect(validSummaries.has(snap.semanticSummary)).toBe(true);
    }
  });
});
