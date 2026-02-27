import { describe, it, expect, vi } from "vitest";
import { SemanticStateEngine, type Snapshot } from "./SemanticStateEngine.js";
import { cosineSimilarity, emaFusion } from "../math/vector.js";

// ── Constants mirrored from Rust crate/src/lib.rs ──────────────────────────────
const AGE_DECAY_RATE = 0.0001;
const DRIFT_WEIGHT = 0.5;

/**
 * Pure TypeScript implementation of the core logic, acting as the "Golden Master"
 * to verify the Rust/WASM behavior.
 */
class ShadowStateEngine {
  private stateVector: number[] = [];
  private lastUpdatedAt: number = 0;
  private lastDrift: number = 0;
  private updateCount: number = 0;

  constructor(
    private readonly alpha: number,
    private readonly driftThreshold: number,
    private readonly onDriftDetected?: (
      vector: number[],
      driftScore: number,
    ) => void,
  ) {}

  update(embedding: number[], nowMs: number) {
    if (embedding.length === 0) {
      throw new Error("Embedding must not be empty");
    }

    if (this.updateCount === 0) {
      const zero = new Array(embedding.length).fill(0);
      this.stateVector = emaFusion(embedding, zero, this.alpha);
      this.lastDrift = 0;
    } else {
      if (embedding.length !== this.stateVector.length) {
        throw new Error(
          `Embedding dimension mismatch: expected ${this.stateVector.length}, got ${embedding.length}`,
        );
      }
      const similarity = cosineSimilarity(this.stateVector, embedding);
      const drift = 1.0 - similarity;
      const detected = similarity < this.driftThreshold;

      if (detected) {
        this.onDriftDetected?.(embedding, drift);
      }

      this.stateVector = emaFusion(embedding, this.stateVector, this.alpha);
      this.lastDrift = drift;
    }

    this.lastUpdatedAt = nowMs;
    this.updateCount++;
  }

  getSnapshot(nowMs: number): Snapshot {
    const healthScore = this.calculateHealth(nowMs);
    return {
      vector: [...this.stateVector],
      healthScore,
      timestamp: this.lastUpdatedAt,
      semanticSummary: this.buildSummary(healthScore),
    };
  }

  private calculateHealth(nowMs: number): number {
    const timeSinceUpdate = Math.max(0, nowMs - this.lastUpdatedAt);
    const agePenalty = timeSinceUpdate * AGE_DECAY_RATE;
    const driftPenalty = this.lastDrift * DRIFT_WEIGHT;
    return Math.max(0, Math.min(1.0, 1.0 - agePenalty - driftPenalty));
  }

  private buildSummary(healthScore: number): string {
    if (healthScore > 0.8) {
      return "stable";
    } else if (healthScore > 0.5) {
      return "drifting";
    } else {
      return "volatile";
    }
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function randomVector(dim: number): number[] {
  return Array.from({ length: dim }, () => Math.random() * 2 - 1);
}

// ── Parity Tests ─────────────────────────────────────────────────────────────

describe("Parity: Rust WASM Engine vs TypeScript Shadow Engine", () => {
  it("should maintain identical state over a sequence of random updates", async () => {
    const DIM = 32;
    const STEPS = 100;
    const ALPHA = 0.5;
    const DRIFT_THRESHOLD = 0.7;

    // We will control time manually
    let currentTime = 10000;

    const driftSpyWasm = vi.fn();
    const driftSpyShadow = vi.fn();

    // 1. Setup WASM Engine
    // We mock the provider because SemanticStateEngine.update() calls provider.getEmbedding()
    // but the actual math happens in WASM. We'll feed it raw vectors via the mock.
    const wasmProvider = {
      getEmbedding: vi.fn(),
    };
    const wasmEngine = new SemanticStateEngine({
      alpha: ALPHA,
      driftThreshold: DRIFT_THRESHOLD,
      provider: wasmProvider as any,
      onDriftDetected: driftSpyWasm,
    });

    // 2. Setup Shadow Engine
    const shadowEngine = new ShadowStateEngine(
      ALPHA,
      DRIFT_THRESHOLD,
      driftSpyShadow,
    );

    // 3. Fuzzing Loop
    for (let i = 0; i < STEPS; i++) {
      const vec = randomVector(DIM);
      const text = `step-${i}`;

      // Advance time randomly
      currentTime += Math.floor(Math.random() * 5000);

      // --- Update WASM Engine ---
      // We need to inject the time. SemanticStateEngine calls Date.now().
      // We'll mock Date.now() to ensure deterministic time sync.
      vi.setSystemTime(currentTime);

      wasmProvider.getEmbedding.mockResolvedValueOnce(new Float32Array(vec));
      await wasmEngine.update(text);

      // --- Update Shadow Engine ---
      shadowEngine.update(vec, currentTime);

      // --- Verify ---
      const snapWasm = wasmEngine.getSnapshot();
      const snapShadow = shadowEngine.getSnapshot(currentTime);

      // A. State Vector
      expect(snapWasm.vector).toHaveLength(DIM);
      snapWasm.vector.forEach((val, idx) => {
        expect(val).toBeCloseTo(snapShadow.vector[idx], 5);
      });

      // B. Health Score
      expect(snapWasm.healthScore).toBeCloseTo(snapShadow.healthScore, 5);

      // C. Semantic Summary
      expect(snapWasm.semanticSummary).toBe(snapShadow.semanticSummary);

      // D. Timestamp
      expect(snapWasm.timestamp).toBe(snapShadow.timestamp);
    }

    // E. Drift Callbacks
    expect(driftSpyWasm).toHaveBeenCalledTimes(driftSpyShadow.mock.calls.length);

    // Verify args for each drift call
    for (let k = 0; k < driftSpyWasm.mock.calls.length; k++) {
        const [vecWasm, scoreWasm] = driftSpyWasm.mock.calls[k];
        const [vecShadow, scoreShadow] = driftSpyShadow.mock.calls[k];

        // Vectors match?
        vecWasm.forEach((val: number, idx: number) => {
            expect(val).toBeCloseTo(vecShadow[idx], 5);
        });

        // Drift scores match?
        expect(scoreWasm).toBeCloseTo(scoreShadow, 5);
    }
  });

  it("should handle edge case: Zero Vector (from empty text)", async () => {
    // Rust cosine_similarity returns 0.0 if magnitude is 0.
    const ALPHA = 0.5;
    const DRIFT_THRESHOLD = 0.5;
    const currentTime = 1000;

    vi.setSystemTime(currentTime);

    const wasmProvider = { getEmbedding: vi.fn() };
    const wasmEngine = new SemanticStateEngine({
        alpha: ALPHA,
        driftThreshold: DRIFT_THRESHOLD,
        provider: wasmProvider as any
    });

    const shadowEngine = new ShadowStateEngine(ALPHA, DRIFT_THRESHOLD);

    const zeroVec = [0, 0, 0, 0];

    // First update (establishes baseline)
    wasmProvider.getEmbedding.mockResolvedValueOnce(new Float32Array(zeroVec));
    await wasmEngine.update("zero");
    shadowEngine.update(zeroVec, currentTime);

    // Check parity
    const wSnap = wasmEngine.getSnapshot();
    const sSnap = shadowEngine.getSnapshot(currentTime);
    expect(wSnap.vector).toEqual(sSnap.vector); // Should be all zeros
  });

  it("should handle drift detection boundary conditions", async () => {
      // Setup identical engines
      const ALPHA = 0.1;
      const DRIFT_THRESHOLD = 0.9; // High threshold, sensitive to drift
      const currentTime = 5000;
      vi.setSystemTime(currentTime);

      const driftSpyWasm = vi.fn();
      const driftSpyShadow = vi.fn();

      const wasmProvider = { getEmbedding: vi.fn() };
      const wasmEngine = new SemanticStateEngine({
          alpha: ALPHA,
          driftThreshold: DRIFT_THRESHOLD,
          provider: wasmProvider as any,
          onDriftDetected: driftSpyWasm
      });
      const shadowEngine = new ShadowStateEngine(ALPHA, DRIFT_THRESHOLD, driftSpyShadow);

      // 1. Baseline
      const vecA = [1, 0];
      wasmProvider.getEmbedding.mockResolvedValueOnce(new Float32Array(vecA));
      await wasmEngine.update("A");
      shadowEngine.update(vecA, currentTime);

      // 2. Small change (no drift)
      // cos([1,0], [0.99, 0.01]) ~ 1.0 > 0.9
      const vecB = [0.99, 0.01];
      wasmProvider.getEmbedding.mockResolvedValueOnce(new Float32Array(vecB));
      await wasmEngine.update("B");
      shadowEngine.update(vecB, currentTime);

      expect(driftSpyWasm).not.toHaveBeenCalled();
      expect(driftSpyShadow).not.toHaveBeenCalled();

      // 3. Large change (drift)
      // cos([1,0], [0, 1]) = 0 < 0.9
      const vecC = [0, 1];
      wasmProvider.getEmbedding.mockResolvedValueOnce(new Float32Array(vecC));
      await wasmEngine.update("C");
      shadowEngine.update(vecC, currentTime);

      expect(driftSpyWasm).toHaveBeenCalledTimes(1);
      expect(driftSpyShadow).toHaveBeenCalledTimes(1);

      const [, driftScoreWasm] = driftSpyWasm.mock.calls[0];
      const [, driftScoreShadow] = driftSpyShadow.mock.calls[0];

      // Drift score = 1 - cos = 1 - 0 = 1
      // Note: Current state might have shifted slightly towards B, so it won't be exactly [1,0]
      expect(driftScoreWasm).toBeCloseTo(driftScoreShadow, 5);
  });
});
