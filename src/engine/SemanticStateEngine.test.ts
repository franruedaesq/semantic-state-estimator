import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { SemanticStateEngine } from "./SemanticStateEngine.js";
import type { WorkerManager } from "../worker/WorkerManager.js";

const DIM = 4;

/**
 * Builds a minimal WorkerManager mock that returns each vector in `vectors`
 * (cycling) as a Float32Array on successive `getEmbedding` calls.
 */
function makeWorkerManager(vectors: number[][]): WorkerManager {
  let callCount = 0;
  return {
    getEmbedding: vi.fn().mockImplementation(() => {
      const vec = vectors[callCount % vectors.length]!;
      callCount++;
      return Promise.resolve(new Float32Array(vec));
    }),
  } as unknown as WorkerManager;
}

function vec(values: number[]): number[] {
  const result = new Array(DIM).fill(0) as number[];
  values.slice(0, DIM).forEach((v, i) => {
    result[i] = v;
  });
  return result;
}

describe("SemanticStateEngine (engine/)", () => {
  // ─── Integration with WorkerManager ─────────────────────────────────────────

  describe("WorkerManager integration", () => {
    it("calls workerManager.getEmbedding with the supplied text", async () => {
      const wm = makeWorkerManager([vec([1, 0, 0, 0])]);
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        workerManager: wm,
      });

      await engine.update("some UI event");

      expect(wm.getEmbedding).toHaveBeenCalledTimes(1);
      expect(wm.getEmbedding).toHaveBeenCalledWith("some UI event");
    });

    it("applies EMA fusion after the worker resolves", async () => {
      // S_0 = [0,0,0,0], E_1 = [1,0,0,0], α = 0.5
      // S_1 = 0.5*[1,0,0,0] + 0.5*[0,0,0,0] = [0.5, 0, 0, 0]
      const wm = makeWorkerManager([vec([1, 0, 0, 0])]);
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        workerManager: wm,
      });

      await engine.update("first event");
      const { vector } = engine.getSnapshot();

      expect(vector[0]).toBeCloseTo(0.5);
      expect(vector[1]).toBeCloseTo(0);
    });

    it("triggers onDriftDetected when the embedding drifts beyond the threshold", async () => {
      const onDriftDetected = vi.fn();
      const wm = makeWorkerManager([vec([1, 0, 0, 0]), vec([0, 1, 0, 0])]);
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        onDriftDetected,
        workerManager: wm,
      });

      await engine.update("first event");
      expect(onDriftDetected).not.toHaveBeenCalled();

      // Orthogonal vector → cosine similarity = 0, well below 0.75.
      await engine.update("second event");
      expect(onDriftDetected).toHaveBeenCalledTimes(1);
    });

    it("updates healthScore after resolving the embedding", async () => {
      vi.useFakeTimers();
      const wm = makeWorkerManager([vec([1, 0, 0, 0])]);
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        workerManager: wm,
      });

      await engine.update("event");
      const { healthScore } = engine.getSnapshot();
      expect(healthScore).toBeCloseTo(1.0);

      vi.useRealTimers();
    });
  });

  // ─── onDriftDetected ────────────────────────────────────────────────────────

  describe("onDriftDetected callback", () => {
    it("fires when cosine similarity drops below driftThreshold", async () => {
      const onDriftDetected = vi.fn();
      const wm = makeWorkerManager([vec([1, 0, 0, 0]), vec([0, 1, 0, 0])]);
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        onDriftDetected,
        workerManager: wm,
      });

      await engine.update("first");
      expect(onDriftDetected).not.toHaveBeenCalled();

      await engine.update("second");
      expect(onDriftDetected).toHaveBeenCalledTimes(1);
      expect(onDriftDetected).toHaveBeenCalledWith(
        vec([0, 1, 0, 0]),
        expect.any(Number),
      );
    });

    it("does NOT fire when cosine similarity stays above driftThreshold", async () => {
      const onDriftDetected = vi.fn();
      const wm = makeWorkerManager([vec([1, 0, 0, 0]), vec([1, 0, 0, 0])]);
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        onDriftDetected,
        workerManager: wm,
      });

      await engine.update("first");
      await engine.update("second");
      expect(onDriftDetected).not.toHaveBeenCalled();
    });

    it("passes the drift score as a number to the callback", async () => {
      const onDriftDetected = vi.fn();
      const wm = makeWorkerManager([vec([1, 0, 0, 0]), vec([0, 1, 0, 0])]);
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        onDriftDetected,
        workerManager: wm,
      });

      await engine.update("first");
      await engine.update("second");

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

    it("starts at 1.0 immediately after the first update", async () => {
      vi.useFakeTimers();
      const wm = makeWorkerManager([vec([1, 0, 0, 0])]);
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        workerManager: wm,
      });
      await engine.update("event");
      const { healthScore } = engine.getSnapshot();
      expect(healthScore).toBeCloseTo(1.0);
    });

    it("decays with age: healthScore decreases as time advances", async () => {
      vi.useFakeTimers();
      const wm = makeWorkerManager([vec([1, 0, 0, 0])]);
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        workerManager: wm,
      });
      await engine.update("event");
      const { healthScore: before } = engine.getSnapshot();

      vi.advanceTimersByTime(5000);

      const { healthScore: after } = engine.getSnapshot();
      expect(after).toBeLessThan(before);
    });

    it("drops with volatility: rapidly shifting vectors lower healthScore", async () => {
      const wm = makeWorkerManager([vec([1, 0, 0, 0]), vec([0, 1, 0, 0])]);
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        workerManager: wm,
      });
      await engine.update("first");
      const { healthScore: stable } = engine.getSnapshot();

      await engine.update("second");
      const { healthScore: volatile } = engine.getSnapshot();

      expect(volatile).toBeLessThan(stable);
    });

    it("is clamped to [0, 1]", async () => {
      vi.useFakeTimers();
      const wm = makeWorkerManager([vec([1, 0, 0, 0])]);
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        workerManager: wm,
      });
      await engine.update("event");

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
      const wm = makeWorkerManager([
        vec([1, 0, 0, 0]),
        vec([0, 1, 0, 0]),
        vec([1, 2, 3, 4]),
      ]);
      engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        workerManager: wm,
      });
    });

    it("applies S_t = α·E_t + (1−α)·S_{t−1} on first update (from zero origin)", async () => {
      // S_0 = [0,0,0,0], E_1 = [1,0,0,0], α = 0.5
      // S_1 = 0.5*[1,0,0,0] + 0.5*[0,0,0,0] = [0.5, 0, 0, 0]
      await engine.update("first");
      const { vector } = engine.getSnapshot();
      expect(vector[0]).toBeCloseTo(0.5);
      expect(vector[1]).toBeCloseTo(0);
    });

    it("applies EMA correctly on the second update", async () => {
      // After first: S_1 = [0.5, 0, 0, 0]
      // E_2 = [0, 1, 0, 0], α = 0.5
      // S_2 = 0.5*[0,1,0,0] + 0.5*[0.5,0,0,0] = [0.25, 0.5, 0, 0]
      await engine.update("first");
      await engine.update("second");
      const { vector } = engine.getSnapshot();
      expect(vector[0]).toBeCloseTo(0.25);
      expect(vector[1]).toBeCloseTo(0.5);
    });

    it("getSnapshot returns a copy, not a reference", async () => {
      await engine.update("first");
      const snap = engine.getSnapshot();
      snap.vector[0] = 999;
      expect(engine.getSnapshot().vector[0]).not.toBe(999);
    });
  });

  // ─── Snapshot shape ──────────────────────────────────────────────────────────

  describe("getSnapshot", () => {
    it("returns a snapshot with all required fields", async () => {
      const wm = makeWorkerManager([vec([1, 0, 0, 0])]);
      const engine = new SemanticStateEngine({
        alpha: 0.5,
        driftThreshold: 0.75,
        workerManager: wm,
      });
      await engine.update("event");
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

