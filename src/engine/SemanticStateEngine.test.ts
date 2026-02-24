import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { SemanticStateEngine } from "./SemanticStateEngine.js";

const DIM = 4;

function vec(values: number[]): number[] {
  const result = new Array(DIM).fill(0) as number[];
  values.slice(0, DIM).forEach((v, i) => {
    result[i] = v;
  });
  return result;
}

describe("SemanticStateEngine (engine/)", () => {
  // ─── onDriftDetected ────────────────────────────────────────────────────────

  describe("onDriftDetected callback", () => {
    it("fires when cosine similarity drops below driftThreshold", () => {
      const onDriftDetected = vi.fn();
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        onDriftDetected,
      });

      // First push sets the baseline — no callback yet.
      engine.update(vec([1, 0, 0, 0]));
      expect(onDriftDetected).not.toHaveBeenCalled();

      // Orthogonal vector → cosine similarity = 0, well below 0.75.
      engine.update(vec([0, 1, 0, 0]));
      expect(onDriftDetected).toHaveBeenCalledTimes(1);
      expect(onDriftDetected).toHaveBeenCalledWith(
        vec([0, 1, 0, 0]),
        expect.any(Number),
      );
    });

    it("does NOT fire when cosine similarity stays above driftThreshold", () => {
      const onDriftDetected = vi.fn();
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        onDriftDetected,
      });

      engine.update(vec([1, 0, 0, 0]));
      // Same direction — similarity = 1, above threshold.
      engine.update(vec([1, 0, 0, 0]));
      expect(onDriftDetected).not.toHaveBeenCalled();
    });

    it("passes the drift score as a number to the callback", () => {
      const onDriftDetected = vi.fn();
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        onDriftDetected,
      });

      engine.update(vec([1, 0, 0, 0]));
      engine.update(vec([0, 1, 0, 0]));

      const [, driftScore] = onDriftDetected.mock.calls[0] as [
        number[],
        number,
      ];
      expect(typeof driftScore).toBe("number");
      // Orthogonal vectors: similarity = 0, so drift = 1 - 0 = 1.
      expect(driftScore).toBeCloseTo(1);
    });
  });

  // ─── healthScore ────────────────────────────────────────────────────────────

  describe("healthScore", () => {
    afterEach(() => {
      vi.useRealTimers();
    });

    it("starts at 1.0 immediately after the first update", () => {
      vi.useFakeTimers();
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
      });
      engine.update(vec([1, 0, 0, 0]));
      const { healthScore } = engine.getSnapshot();
      expect(healthScore).toBeCloseTo(1.0);
    });

    it("decays with age: healthScore decreases as time advances", () => {
      vi.useFakeTimers();
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
      });
      engine.update(vec([1, 0, 0, 0]));
      const { healthScore: before } = engine.getSnapshot();

      vi.advanceTimersByTime(5000); // advance 5 seconds

      const { healthScore: after } = engine.getSnapshot();
      expect(after).toBeLessThan(before);
    });

    it("drops with volatility: rapidly shifting vectors lower healthScore", () => {
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
      });
      engine.update(vec([1, 0, 0, 0]));
      const { healthScore: stable } = engine.getSnapshot();

      // Push an orthogonal vector to introduce maximum drift.
      engine.update(vec([0, 1, 0, 0]));
      const { healthScore: volatile } = engine.getSnapshot();

      expect(volatile).toBeLessThan(stable);
    });

    it("is clamped to [0, 1]", () => {
      vi.useFakeTimers();
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
      });
      engine.update(vec([1, 0, 0, 0]));

      // Advance far into the future so age penalty saturates.
      vi.advanceTimersByTime(60_000);

      const { healthScore } = engine.getSnapshot();
      expect(healthScore).toBeGreaterThanOrEqual(0);
      expect(healthScore).toBeLessThanOrEqual(1);
    });
  });

  // ─── EMA fusion ─────────────────────────────────────────────────────────────

  describe("EMA fusion", () => {
    let engine: SemanticStateEngine;

    beforeEach(() => {
      engine = new SemanticStateEngine({ alpha: 0.5, driftThreshold: 0.75 });
    });

    it("applies S_t = α·E_t + (1−α)·S_{t−1} on first update (from zero origin)", () => {
      // S_0 = [0,0,0,0], E_1 = [1,0,0,0], α = 0.5
      // S_1 = 0.5*[1,0,0,0] + 0.5*[0,0,0,0] = [0.5, 0, 0, 0]
      engine.update(vec([1, 0, 0, 0]));
      const { vector } = engine.getSnapshot();
      expect(vector[0]).toBeCloseTo(0.5);
      expect(vector[1]).toBeCloseTo(0);
    });

    it("applies EMA correctly on the second update", () => {
      // After first: S_1 = [0.5, 0, 0, 0]
      // E_2 = [0, 1, 0, 0], α = 0.5
      // S_2 = 0.5*[0,1,0,0] + 0.5*[0.5,0,0,0] = [0.25, 0.5, 0, 0]
      engine.update(vec([1, 0, 0, 0]));
      engine.update(vec([0, 1, 0, 0]));
      const { vector } = engine.getSnapshot();
      expect(vector[0]).toBeCloseTo(0.25);
      expect(vector[1]).toBeCloseTo(0.5);
    });

    it("getSnapshot returns a copy, not a reference", () => {
      engine.update(vec([1, 0, 0, 0]));
      const snap = engine.getSnapshot();
      snap.vector[0] = 999;
      expect(engine.getSnapshot().vector[0]).not.toBe(999);
    });

    it("throws when a subsequent embedding has a different dimension", () => {
      engine.update(vec([1, 0, 0, 0]));
      expect(() => engine.update([1, 2])).toThrow(
        "Embedding dimension mismatch",
      );
    });
  });

  // ─── Snapshot shape ──────────────────────────────────────────────────────────

  describe("getSnapshot", () => {
    it("returns a snapshot with all required fields", () => {
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
      });
      engine.update(vec([1, 0, 0, 0]));
      const snap = engine.getSnapshot();

      expect(snap).toHaveProperty("vector");
      expect(snap).toHaveProperty("healthScore");
      expect(snap).toHaveProperty("timestamp");
      expect(snap).toHaveProperty("semanticSummary");
      expect(typeof snap.semanticSummary).toBe("string");
      expect(typeof snap.timestamp).toBe("number");
    });
  });
});
