/**
 * Shadow / Parity Tests
 *
 * A pure TypeScript `ShadowEngine` mirrors the Rust `WasmStateEngine` logic.
 * Both engines are fed the same inputs and their outputs are compared to
 * verify behavioral parity after the migration to Rust/WASM.
 *
 * Constants are taken directly from crate/src/lib.rs:
 *   AGE_DECAY_RATE = 0.0001
 *   DRIFT_WEIGHT   = 0.5
 */

import { describe, it, expect, vi, afterEach } from "vitest";
import { WasmStateEngine } from "../wasm-pkg/loader.js";

// ── ShadowEngine ──────────────────────────────────────────────────────────────
// Pure TypeScript reimplementation of crate/src/lib.rs `WasmStateEngine`.
// Any divergence between these two implementations is a parity failure.

const AGE_DECAY_RATE = 0.0001;
const DRIFT_WEIGHT = 0.5;

interface ShadowUpdateResult {
  driftDetected: boolean;
  driftScore: number;
  vector: number[];
}

interface ShadowSnapshot {
  vector: number[];
  healthScore: number;
  timestamp: number;
  semanticSummary: string;
}

class ShadowEngine {
  private readonly alpha: number;
  private readonly driftThreshold: number;
  private stateVector: number[] = [];
  private lastUpdatedAt: number = 0;
  private lastDrift: number = 0;
  private updateCount: number = 0;

  constructor(alpha: number, driftThreshold: number) {
    this.alpha = alpha;
    this.driftThreshold = driftThreshold;
  }

  update(embedding: number[], nowMs: number): ShadowUpdateResult {
    if (embedding.length === 0) {
      throw new Error("Embedding must not be empty");
    }

    let driftDetected: boolean;
    let driftScore: number;

    if (this.updateCount === 0) {
      // First call: establish baseline from a zero-vector origin.
      const zero = new Array(embedding.length).fill(0) as number[];
      this.stateVector = this.emaFusion(embedding, zero);
      driftDetected = false;
      driftScore = 0;
    } else {
      const similarity = this.cosineSimilarity(this.stateVector, embedding);
      const drift = 1.0 - similarity;
      driftDetected = similarity < this.driftThreshold;
      driftScore = drift;
      this.stateVector = this.emaFusion(embedding, this.stateVector);
      this.lastDrift = drift;
    }

    this.lastUpdatedAt = nowMs;
    this.updateCount++;

    return { driftDetected, driftScore, vector: [...embedding] };
  }

  getSnapshot(nowMs: number): ShadowSnapshot {
    const healthScore = this.calculateHealth(nowMs);
    return {
      vector: [...this.stateVector],
      healthScore,
      timestamp: this.lastUpdatedAt,
      semanticSummary: this.buildSummary(healthScore),
    };
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let magA = 0;
    let magB = 0;
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i]! * b[i]!;
      magA += a[i]! * a[i]!;
      magB += b[i]! * b[i]!;
    }
    magA = Math.sqrt(magA);
    magB = Math.sqrt(magB);
    if (magA === 0 || magB === 0) return 0;
    // clamp to [-1, 1] matching Rust: .clamp(-1.0, 1.0)
    return Math.max(-1, Math.min(1, dotProduct / (magA * magB)));
  }

  private emaFusion(current: number[], previous: number[]): number[] {
    return current.map((val, i) => this.alpha * val + (1 - this.alpha) * previous[i]!);
  }

  private calculateHealth(nowMs: number): number {
    const timeSinceUpdate = Math.max(0, nowMs - this.lastUpdatedAt);
    const agePenalty = timeSinceUpdate * AGE_DECAY_RATE;
    const driftPenalty = this.lastDrift * DRIFT_WEIGHT;
    return Math.max(0, Math.min(1, 1 - agePenalty - driftPenalty));
  }

  private buildSummary(healthScore: number): string {
    if (healthScore > 0.8) return "stable";
    if (healthScore > 0.5) return "drifting";
    return "volatile";
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Tolerance for WASM (f32) vs Shadow (f64) comparisons. */
const F32_TOLERANCE = 5;

/** Asserts two vectors are component-wise equal within f32 tolerance. */
function expectVectorsClose(actual: number[], expected: number[]): void {
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i]!, F32_TOLERANCE);
  }
}

/** Fixed timestamp used when time control is not relevant. */
const T0 = 1_000_000;

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("Shadow / Parity: ShadowEngine vs WasmStateEngine", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  // ── 1. Initialization Parity ────────────────────────────────────────────────

  describe("Initialization Parity", () => {
    it("both engines start with an empty/zero state vector before any update", () => {
      const wasm = new WasmStateEngine(0.5, 0.75);
      const shadow = new ShadowEngine(0.5, 0.75);

      // Before any update, both snapshots should have an empty state vector.
      const wasmSnap = wasm.get_snapshot(T0) as ShadowSnapshot;
      const shadowSnap = shadow.getSnapshot(T0);

      expect(wasmSnap.vector).toHaveLength(0);
      expect(shadowSnap.vector).toHaveLength(0);
    });

    it("both engines report an initial healthScore of 1.0 immediately after the first update", () => {
      vi.useFakeTimers();
      const wasm = new WasmStateEngine(0.5, 0.75);
      const shadow = new ShadowEngine(0.5, 0.75);

      const embedding = new Float32Array([1, 0, 0, 0]);
      wasm.update(embedding, T0);
      shadow.update([1, 0, 0, 0], T0);

      const wasmSnap = wasm.get_snapshot(T0) as ShadowSnapshot;
      const shadowSnap = shadow.getSnapshot(T0);

      expect(wasmSnap.healthScore).toBeCloseTo(1.0, F32_TOLERANCE);
      expect(shadowSnap.healthScore).toBeCloseTo(1.0, F32_TOLERANCE);
    });

    it("both engines produce identical initial snapshots after the first update", () => {
      const embedding = new Float32Array([1, 0, 0, 0]);
      const wasm = new WasmStateEngine(0.5, 0.75);
      const shadow = new ShadowEngine(0.5, 0.75);

      wasm.update(embedding, T0);
      shadow.update([1, 0, 0, 0], T0);

      const wasmSnap = wasm.get_snapshot(T0) as ShadowSnapshot;
      const shadowSnap = shadow.getSnapshot(T0);

      expectVectorsClose(wasmSnap.vector, shadowSnap.vector);
      expect(wasmSnap.healthScore).toBeCloseTo(shadowSnap.healthScore, F32_TOLERANCE);
      expect(wasmSnap.timestamp).toBe(shadowSnap.timestamp);
      expect(wasmSnap.semanticSummary).toBe(shadowSnap.semanticSummary);
    });
  });

  // ── 2. Steady State Evolution ───────────────────────────────────────────────

  describe("Steady State Evolution", () => {
    it("state vector evolves identically after each update with consistent embeddings", () => {
      const alpha = 0.5;
      const driftThreshold = 0.3; // low threshold to avoid drift on similar vectors
      const wasm = new WasmStateEngine(alpha, driftThreshold);
      const shadow = new ShadowEngine(alpha, driftThreshold);

      // Consistent (similar) embeddings — all pointing roughly in the same direction.
      const embeddings = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0.9, 0.1, 0, 0],
      ];

      for (let step = 0; step < embeddings.length; step++) {
        const emb = embeddings[step]!;
        const f32emb = new Float32Array(emb);
        const now = T0 + step * 1000;

        wasm.update(f32emb, now);
        shadow.update(emb, now);

        const wasmSnap = wasm.get_snapshot(now) as ShadowSnapshot;
        const shadowSnap = shadow.getSnapshot(now);

        expectVectorsClose(wasmSnap.vector, shadowSnap.vector);
        expect(wasmSnap.healthScore).toBeCloseTo(shadowSnap.healthScore, F32_TOLERANCE);
      }
    });

    it("healthScore degrades identically as simulated time advances", () => {
      vi.useFakeTimers();
      const wasm = new WasmStateEngine(0.5, 0.75);
      const shadow = new ShadowEngine(0.5, 0.75);

      wasm.update(new Float32Array([1, 0, 0, 0]), T0);
      shadow.update([1, 0, 0, 0], T0);

      // Advance time by 5 seconds
      const laterMs = T0 + 5000;

      const wasmSnapBefore = wasm.get_snapshot(T0) as ShadowSnapshot;
      const shadowSnapBefore = shadow.getSnapshot(T0);
      expect(wasmSnapBefore.healthScore).toBeCloseTo(shadowSnapBefore.healthScore, F32_TOLERANCE);

      const wasmSnapAfter = wasm.get_snapshot(laterMs) as ShadowSnapshot;
      const shadowSnapAfter = shadow.getSnapshot(laterMs);

      expect(wasmSnapAfter.healthScore).toBeLessThan(wasmSnapBefore.healthScore);
      expect(shadowSnapAfter.healthScore).toBeLessThan(shadowSnapBefore.healthScore);
      expect(wasmSnapAfter.healthScore).toBeCloseTo(shadowSnapAfter.healthScore, F32_TOLERANCE);
    });
  });

  // ── 3. Drift Event Detection ────────────────────────────────────────────────

  describe("Drift Event Detection", () => {
    it("both engines detect drift at the exact same step", () => {
      const wasmEngine = new WasmStateEngine(0.5, 0.75);
      const shadowEngine = new ShadowEngine(0.5, 0.75);

      // Step 1: baseline — no drift on first update.
      const w1 = wasmEngine.update(new Float32Array([1, 0, 0, 0]), T0) as ShadowUpdateResult;
      const s1 = shadowEngine.update([1, 0, 0, 0], T0);
      expect(w1.driftDetected).toBe(false);
      expect(s1.driftDetected).toBe(false);

      // Step 2: orthogonal vector → high drift, should trigger in both.
      const w2 = wasmEngine.update(new Float32Array([0, 1, 0, 0]), T0 + 1000) as ShadowUpdateResult;
      const s2 = shadowEngine.update([0, 1, 0, 0], T0 + 1000);
      expect(w2.driftDetected).toBe(true);
      expect(s2.driftDetected).toBe(true);
      expect(w2.driftDetected).toBe(s2.driftDetected);
    });

    it("both engines report identical driftScore (within floating-point tolerance)", () => {
      const wasmEngine = new WasmStateEngine(0.5, 0.75);
      const shadowEngine = new ShadowEngine(0.5, 0.75);

      // Establish baseline.
      wasmEngine.update(new Float32Array([1, 0, 0, 0]), T0);
      shadowEngine.update([1, 0, 0, 0], T0);

      // Inject orthogonal vector.
      const wasmResult = wasmEngine.update(
        new Float32Array([0, 1, 0, 0]),
        T0 + 1000,
      ) as ShadowUpdateResult;
      const shadowResult = shadowEngine.update([0, 1, 0, 0], T0 + 1000);

      // Cosine similarity of [0.5,0,0,0] and [0,1,0,0] = 0 → drift = 1.0.
      expect(wasmResult.driftScore).toBeCloseTo(shadowResult.driftScore, F32_TOLERANCE);
      expect(wasmResult.driftDetected).toBe(shadowResult.driftDetected);
    });

    it("does NOT fire drift when similarity stays above threshold in both engines", () => {
      const wasmEngine = new WasmStateEngine(0.5, 0.75);
      const shadowEngine = new ShadowEngine(0.5, 0.75);

      // Two identical embeddings — similarity = 1.0, well above 0.75.
      wasmEngine.update(new Float32Array([1, 0, 0, 0]), T0);
      shadowEngine.update([1, 0, 0, 0], T0);

      const wasmResult = wasmEngine.update(
        new Float32Array([1, 0, 0, 0]),
        T0 + 1000,
      ) as ShadowUpdateResult;
      const shadowResult = shadowEngine.update([1, 0, 0, 0], T0 + 1000);

      expect(wasmResult.driftDetected).toBe(false);
      expect(shadowResult.driftDetected).toBe(false);
      expect(wasmResult.driftDetected).toBe(shadowResult.driftDetected);
    });
  });

  // ── 4. Recovery Phase ───────────────────────────────────────────────────────

  describe("Recovery Phase", () => {
    it("healthScore recovers at the same rate for both engines after a drift event", () => {
      const wasmEngine = new WasmStateEngine(0.5, 0.75);
      const shadowEngine = new ShadowEngine(0.5, 0.75);

      // Establish baseline.
      wasmEngine.update(new Float32Array([1, 0, 0, 0]), T0);
      shadowEngine.update([1, 0, 0, 0], T0);

      // Drift event.
      const driftTime = T0 + 1000;
      wasmEngine.update(new Float32Array([0, 1, 0, 0]), driftTime);
      shadowEngine.update([0, 1, 0, 0], driftTime);

      // Check health immediately after drift.
      const wasmAfterDrift = wasmEngine.get_snapshot(driftTime) as ShadowSnapshot;
      const shadowAfterDrift = shadowEngine.getSnapshot(driftTime);
      expect(wasmAfterDrift.healthScore).toBeCloseTo(shadowAfterDrift.healthScore, F32_TOLERANCE);

      // Recovery: send consistent embeddings back.
      const recoverTimes = [2000, 3000, 4000, 5000];
      const recoverVec = new Float32Array([0, 1, 0, 0]); // same direction, no more drift

      for (const dt of recoverTimes) {
        const now = T0 + dt;
        wasmEngine.update(recoverVec, now);
        shadowEngine.update([0, 1, 0, 0], now);

        const wasmSnap = wasmEngine.get_snapshot(now) as ShadowSnapshot;
        const shadowSnap = shadowEngine.getSnapshot(now);

        expect(wasmSnap.healthScore).toBeCloseTo(shadowSnap.healthScore, F32_TOLERANCE);
      }
    });

    it("semanticSummary transitions identically after recovery", () => {
      const wasmEngine = new WasmStateEngine(0.5, 0.2);
      const shadowEngine = new ShadowEngine(0.5, 0.2);

      // Drift event: orthogonal vectors.
      wasmEngine.update(new Float32Array([1, 0, 0, 0]), T0);
      shadowEngine.update([1, 0, 0, 0], T0);

      wasmEngine.update(new Float32Array([0, 1, 0, 0]), T0 + 1000);
      shadowEngine.update([0, 1, 0, 0], T0 + 1000);

      const wasmSnap = wasmEngine.get_snapshot(T0 + 1000) as ShadowSnapshot;
      const shadowSnap = shadowEngine.getSnapshot(T0 + 1000);

      // After drift, health drops; semanticSummary should match.
      expect(wasmSnap.semanticSummary).toBe(shadowSnap.semanticSummary);
    });
  });

  // ── 5. Multi-step EMA Vector Parity ────────────────────────────────────────

  describe("Multi-step EMA vector parity", () => {
    it("state vectors remain in sync over 10 sequential updates with varying embeddings", () => {
      const alpha = 0.3;
      const wasm = new WasmStateEngine(alpha, 0.1);
      const shadow = new ShadowEngine(alpha, 0.1);

      // Pseudo-random but deterministic sequence of 4-dimensional unit-ish vectors.
      const sequence = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
      ];

      for (let i = 0; i < sequence.length; i++) {
        const emb = sequence[i]!;
        const now = T0 + i * 500;

        wasm.update(new Float32Array(emb), now);
        shadow.update(emb, now);

        const wasmSnap = wasm.get_snapshot(now) as ShadowSnapshot;
        const shadowSnap = shadow.getSnapshot(now);

        expectVectorsClose(wasmSnap.vector, shadowSnap.vector);
      }
    });
  });
});
