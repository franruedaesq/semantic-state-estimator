import { describe, it, expect } from "vitest";
import {
  add,
  scale,
  cosineSimilarity,
  emaFusion,
  normalize,
} from "./vector.js";

describe("add", () => {
  it("adds two vectors element-wise", () => {
    expect(add([1, 2, 3], [4, 5, 6])).toEqual([5, 7, 9]);
  });

  it("works with negative values", () => {
    expect(add([-1, 0], [1, 2])).toEqual([0, 2]);
  });

  it("works with zero vectors", () => {
    expect(add([0, 0, 0], [0, 0, 0])).toEqual([0, 0, 0]);
  });

  it("throws on dimension mismatch", () => {
    expect(() => add([1, 2], [1, 2, 3])).toThrow("Vector dimension mismatch");
  });
});

describe("scale", () => {
  it("multiplies every element by a scalar", () => {
    expect(scale([1, 2, 3], 2)).toEqual([2, 4, 6]);
  });

  it("returns zero vector when scalar is 0", () => {
    expect(scale([5, 10], 0)).toEqual([0, 0]);
  });

  it("handles negative scalar", () => {
    expect(scale([1, -2, 3], -1)).toEqual([-1, 2, -3]);
  });

  it("leaves vector unchanged when scalar is 1", () => {
    expect(scale([3, 7], 1)).toEqual([3, 7]);
  });
});

describe("cosineSimilarity", () => {
  it("returns 1 for identical non-zero vectors", () => {
    const v = [1, 2, 3];
    expect(cosineSimilarity(v, v)).toBeCloseTo(1);
  });

  it("returns -1 for opposite vectors", () => {
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1);
  });

  it("returns 0 for orthogonal vectors", () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
  });

  it("returns 0 when a vector has zero magnitude", () => {
    expect(cosineSimilarity([0, 0], [1, 2])).toBe(0);
    expect(cosineSimilarity([1, 2], [0, 0])).toBe(0);
  });

  it("throws on dimension mismatch", () => {
    expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrow(
      "Vector dimension mismatch",
    );
  });
});

describe("normalize", () => {
  it("produces a unit vector", () => {
    const result = normalize([3, 4]);
    const mag = Math.sqrt(result.reduce((s, x) => s + x * x, 0));
    expect(mag).toBeCloseTo(1);
    expect(result[0]).toBeCloseTo(0.6);
    expect(result[1]).toBeCloseTo(0.8);
  });

  it("handles a zero vector without throwing", () => {
    expect(normalize([0, 0, 0])).toEqual([0, 0, 0]);
  });
});

describe("emaFusion", () => {
  it("applies EMA formula: α * E_t + (1 - α) * S_{t-1}", () => {
    const current = [1, 0, 0];
    const previous = [0, 1, 0];
    const result = emaFusion(current, previous, 0.5);
    expect(result).toEqual([0.5, 0.5, 0]);
  });

  it("with alpha=1 returns the current vector exactly", () => {
    expect(emaFusion([3, 1, 4], [0, 0, 0], 1)).toEqual([3, 1, 4]);
  });

  it("alpha=0.1 heavily weights the previous state (slow drift)", () => {
    const result = emaFusion([10, 10], [1, 1], 0.1);
    // 0.1*10 + 0.9*1 = 1.9
    expect(result[0]).toBeCloseTo(1.9);
    expect(result[1]).toBeCloseTo(1.9);
  });

  it("alpha=0.9 heavily weights the current embedding (fast drift)", () => {
    const result = emaFusion([10, 10], [1, 1], 0.9);
    // 0.9*10 + 0.1*1 = 9.1
    expect(result[0]).toBeCloseTo(9.1);
    expect(result[1]).toBeCloseTo(9.1);
  });

  it("low alpha produces smaller drift than high alpha toward a new vector", () => {
    const current = [1, 0];
    const previous = [0, 1];
    const slowResult = emaFusion(current, previous, 0.1);
    const fastResult = emaFusion(current, previous, 0.9);
    // slow result should be closer to previous; fast result closer to current
    expect(slowResult[0]).toBeLessThan(fastResult[0]); // x-axis moves less for slow
    expect(slowResult[1]).toBeGreaterThan(fastResult[1]); // y-axis stays higher for slow
  });

  it("throws on dimension mismatch", () => {
    expect(() => emaFusion([1, 2], [1, 2, 3], 0.5)).toThrow(
      "Vector dimension mismatch",
    );
  });

  it("throws when alpha is out of range (0, 1]", () => {
    expect(() => emaFusion([1], [1], 0)).toThrow("Alpha must be in the range");
    expect(() => emaFusion([1], [1], 1.1)).toThrow(
      "Alpha must be in the range",
    );
  });
});
