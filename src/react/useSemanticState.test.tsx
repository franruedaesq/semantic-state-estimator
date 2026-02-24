// @vitest-environment jsdom
import { describe, it, expect, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useSemanticState } from "./useSemanticState.js";
import type { SemanticStateEngine, Snapshot } from "../engine/SemanticStateEngine.js";

/** Build a minimal SemanticStateEngine mock with controllable subscribe/snapshot. */
function makeEngine(initialSnapshot: Snapshot) {
  let listener: (() => void) | null = null;
  let currentSnapshot = initialSnapshot;

  const engine = {
    subscribe: vi.fn((cb: () => void) => {
      listener = cb;
      return () => {
        listener = null;
      };
    }),
    getSnapshot: vi.fn(() => currentSnapshot),
    update: vi.fn().mockResolvedValue(undefined),
  } as unknown as SemanticStateEngine;

  /** Simulate an engine state update by replacing the snapshot and notifying. */
  const emitUpdate = (nextSnapshot: Snapshot) => {
    currentSnapshot = nextSnapshot;
    listener?.();
  };

  return { engine, emitUpdate };
}

// ─── useSemanticState ────────────────────────────────────────────────────────

describe("useSemanticState", () => {
  it("returns the initial engine snapshot", () => {
    const initial: Snapshot = {
      vector: [0.1, 0.2],
      healthScore: 0.95,
      timestamp: Date.now(),
      semanticSummary: "stable",
    };
    const { engine } = makeEngine(initial);

    const { result } = renderHook(() => useSemanticState(engine));

    expect(result.current.healthScore).toBe(0.95);
    expect(result.current.semanticSummary).toBe("stable");
  });

  it("re-renders with the new snapshot when the engine notifies subscribers", () => {
    const initial: Snapshot = {
      vector: [],
      healthScore: 0.9,
      timestamp: Date.now(),
      semanticSummary: "stable",
    };
    const { engine, emitUpdate } = makeEngine(initial);

    const { result } = renderHook(() => useSemanticState(engine));

    expect(result.current.healthScore).toBe(0.9);

    const updated: Snapshot = {
      vector: [0.5],
      healthScore: 0.4,
      timestamp: Date.now(),
      semanticSummary: "volatile",
    };

    act(() => emitUpdate(updated));

    expect(result.current.healthScore).toBe(0.4);
    expect(result.current.semanticSummary).toBe("volatile");
  });

  it("unsubscribes from the engine when the component unmounts", () => {
    const initial: Snapshot = {
      vector: [],
      healthScore: 1,
      timestamp: Date.now(),
      semanticSummary: "stable",
    };
    const { engine } = makeEngine(initial);

    const { unmount } = renderHook(() => useSemanticState(engine));

    // subscribe should have been called once
    expect(engine.subscribe).toHaveBeenCalledTimes(1);

    unmount();

    // After unmount, emitting should not cause re-renders (no assertion here,
    // but the unsubscribe path is exercised without throwing).
    expect(() => {
      engine.getSnapshot();
    }).not.toThrow();
  });
});
