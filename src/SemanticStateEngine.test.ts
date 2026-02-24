import { describe, it, expect, beforeEach } from "vitest";
import { SemanticStateEngine } from "./SemanticStateEngine.js";

const DIM = 4;

function makeEmbedding(values: number[]): number[] {
  // pad or truncate to DIM
  const result = new Array(DIM).fill(0) as number[];
  values.slice(0, DIM).forEach((v, i) => {
    result[i] = v;
  });
  return result;
}

describe("SemanticStateEngine", () => {
  let engine: SemanticStateEngine;

  beforeEach(() => {
    engine = new SemanticStateEngine({ alpha: 0.5, embeddingDimension: DIM });
  });

  it("initializes with a zero state vector and updateCount of 0", () => {
    const snap = engine.getSnapshot();
    expect(snap.stateVector).toEqual([0, 0, 0, 0]);
    expect(snap.updateCount).toBe(0);
  });

  it("updates state on first embedding regardless of threshold", () => {
    const embedding = makeEmbedding([1, 0, 0, 0]);
    const updated = engine.update(embedding);
    expect(updated).toBe(true);
    expect(engine.getSnapshot().updateCount).toBe(1);
  });

  it("applies EMA correctly on update", () => {
    // First update: state was [0,0,0,0], embedding [1,0,0,0], alpha=0.5
    // S_1 = 0.5*[1,0,0,0] + 0.5*[0,0,0,0] = [0.5, 0, 0, 0]
    engine.update(makeEmbedding([1, 0, 0, 0]));
    let snap = engine.getSnapshot();
    expect(snap.stateVector[0]).toBeCloseTo(0.5);
    expect(snap.stateVector[1]).toBeCloseTo(0);

    // Second update: embedding [1,0,0,0], same direction, should be skipped by threshold
    // cosine sim between [0.5,0,0,0] and [1,0,0,0] = 1.0 → skip
    const updated = engine.update(makeEmbedding([1, 0, 0, 0]));
    expect(updated).toBe(false);
    expect(engine.getSnapshot().updateCount).toBe(1);
  });

  it("updates state when embedding direction changes sufficiently", () => {
    engine.update(makeEmbedding([1, 0, 0, 0]));
    // Orthogonal embedding — cosine similarity = 0, well below threshold
    const updated = engine.update(makeEmbedding([0, 1, 0, 0]));
    expect(updated).toBe(true);
    expect(engine.getSnapshot().updateCount).toBe(2);
  });

  it("getSnapshot returns a copy, not a reference", () => {
    engine.update(makeEmbedding([1, 0, 0, 0]));
    const snap = engine.getSnapshot();
    snap.stateVector[0] = 999;
    expect(engine.getSnapshot().stateVector[0]).not.toBe(999);
  });

  it("reset restores the engine to initial state", () => {
    engine.update(makeEmbedding([1, 0, 0, 0]));
    engine.reset();
    const snap = engine.getSnapshot();
    expect(snap.stateVector).toEqual([0, 0, 0, 0]);
    expect(snap.updateCount).toBe(0);
  });

  it("throws when embedding dimension does not match", () => {
    expect(() => engine.update([1, 2])).toThrow("Embedding dimension mismatch");
  });

  it("throws on invalid alpha", () => {
    expect(
      () => new SemanticStateEngine({ alpha: 0, embeddingDimension: DIM }),
    ).toThrow("alpha must be in (0, 1]");
    expect(
      () => new SemanticStateEngine({ alpha: 1.5, embeddingDimension: DIM }),
    ).toThrow("alpha must be in (0, 1]");
  });

  it("throws on invalid changeThreshold", () => {
    expect(
      () =>
        new SemanticStateEngine({ changeThreshold: -0.1, embeddingDimension: DIM }),
    ).toThrow("changeThreshold must be in [0, 1]");
  });

  it("throws on invalid embeddingDimension", () => {
    expect(
      () => new SemanticStateEngine({ embeddingDimension: 0 }),
    ).toThrow("embeddingDimension must be positive");
  });

  it("lastUpdatedAt advances after an update", async () => {
    const before = engine.getSnapshot().lastUpdatedAt;
    await new Promise((r) => setTimeout(r, 10));
    engine.update(makeEmbedding([1, 0, 0, 0]));
    const after = engine.getSnapshot().lastUpdatedAt;
    expect(new Date(after).getTime()).toBeGreaterThanOrEqual(
      new Date(before).getTime(),
    );
  });
});
