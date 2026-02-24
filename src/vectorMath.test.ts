import { describe, it, expect } from "vitest";
import { emaFusion, cosineSimilarity, normalize } from "./vectorMath.js";

describe("emaFusion", () => {
  it("applies EMA formula: α * E_t + (1 - α) * S_{t-1}", () => {
    const current = [1, 0, 0];
    const previous = [0, 1, 0];
    const alpha = 0.5;
    const result = emaFusion(current, previous, alpha);
    expect(result).toEqual([0.5, 0.5, 0]);
  });

  it("with alpha=1 returns the current vector exactly", () => {
    const current = [3, 1, 4];
    const previous = [0, 0, 0];
    const result = emaFusion(current, previous, 1);
    expect(result).toEqual([3, 1, 4]);
  });

  it("with alpha close to 0 heavily weights the previous state", () => {
    const current = [10, 10];
    const previous = [1, 1];
    const alpha = 0.1;
    const result = emaFusion(current, previous, alpha);
    // 0.1*10 + 0.9*1 = 1.9
    expect(result[0]).toBeCloseTo(1.9);
    expect(result[1]).toBeCloseTo(1.9);
  });

  it("throws when vectors have different dimensions", () => {
    expect(() => emaFusion([1, 2], [1, 2, 3], 0.5)).toThrow(
      "Vector dimension mismatch",
    );
  });

  it("throws when alpha is out of range", () => {
    expect(() => emaFusion([1], [1], 0)).toThrow("Alpha must be in the range");
    expect(() => emaFusion([1], [1], 1.1)).toThrow("Alpha must be in the range");
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

  it("throws when vectors have different dimensions", () => {
    expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrow(
      "Vector dimension mismatch",
    );
  });
});

describe("normalize", () => {
  it("produces a unit vector", () => {
    const v = [3, 4];
    const result = normalize(v);
    const magnitude = Math.sqrt(result.reduce((s, x) => s + x * x, 0));
    expect(magnitude).toBeCloseTo(1);
    expect(result[0]).toBeCloseTo(0.6);
    expect(result[1]).toBeCloseTo(0.8);
  });

  it("handles a zero vector without throwing", () => {
    const result = normalize([0, 0, 0]);
    expect(result).toEqual([0, 0, 0]);
  });

  it("normalizes a single-element vector to [1] or [-1]", () => {
    expect(normalize([5])[0]).toBeCloseTo(1);
    expect(normalize([-5])[0]).toBeCloseTo(-1);
  });
});
