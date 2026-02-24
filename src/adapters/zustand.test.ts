import { describe, it, expect, vi, beforeEach } from "vitest";
import { create } from "zustand";
import { semanticMiddleware } from "./zustand.js";
import type { SemanticMapper } from "./zustand.js";
import type { SemanticStateEngine } from "../engine/SemanticStateEngine.js";

/** Build a minimal SemanticStateEngine mock with a spy on `update`. */
function makeEngine(): SemanticStateEngine {
  return {
    update: vi.fn().mockResolvedValue(undefined),
    subscribe: vi.fn().mockReturnValue(() => undefined),
    getSnapshot: vi.fn().mockReturnValue({
      vector: [],
      healthScore: 1,
      timestamp: Date.now(),
      semanticSummary: "stable",
    }),
  } as unknown as SemanticStateEngine;
}

// ─── Mapper & Lazy Filtering ─────────────────────────────────────────────────

describe("semanticMiddleware", () => {
  let engine: SemanticStateEngine;

  beforeEach(() => {
    engine = makeEngine();
  });

  it("calls engine.update with the string returned by the mapper", async () => {
    type CartStore = { count: number; increment: () => void };
    const mapper: SemanticMapper<CartStore> = (state) =>
      state.count > 0 ? "User added item to cart" : null;

    const useStore = create<CartStore>(
      semanticMiddleware(engine, mapper, (set) => ({
        count: 0,
        increment: () => set((s) => ({ count: s.count + 1 })),
      })),
    );

    useStore.getState().increment();

    // Give the microtask queue a chance to flush.
    await Promise.resolve();

    expect(engine.update).toHaveBeenCalledTimes(1);
    expect(engine.update).toHaveBeenCalledWith("User added item to cart");
  });

  it("does not call engine.update when the mapper returns null (lazy evaluation)", async () => {
    type TextStore = { text: string; setText: (text: string) => void };
    // A mapper that always returns null simulates a noisy event (e.g. keypress).
    const mapper: SemanticMapper<TextStore> = () => null;

    const useStore = create<TextStore>(
      semanticMiddleware(engine, mapper, (set) => ({
        text: "",
        setText: (text: string) => set({ text }),
      })),
    );

    useStore.getState().setText("a");

    await Promise.resolve();

    expect(engine.update).not.toHaveBeenCalled();
  });

  it("does not call engine.update when the mapper returns undefined (lazy evaluation)", async () => {
    type ValueStore = { value: number; setValue: (v: number) => void };
    const mapper: SemanticMapper<ValueStore> = () => undefined;

    const useStore = create<ValueStore>(
      semanticMiddleware(engine, mapper, (set) => ({
        value: 0,
        setValue: (value: number) => set({ value }),
      })),
    );

    useStore.getState().setValue(42);

    await Promise.resolve();

    expect(engine.update).not.toHaveBeenCalled();
  });

  it("passes both next and previous state to the mapper", async () => {
    type CountStore = { count: number; increment: () => void };
    const mapper = vi.fn(
      (_state: CountStore, _prev: CountStore) => null as string | null,
    );

    const useStore = create<CountStore>(
      semanticMiddleware(engine, mapper, (set) => ({
        count: 0,
        increment: () => set((s) => ({ count: s.count + 1 })),
      })),
    );

    useStore.getState().increment();

    await Promise.resolve();

    expect(mapper).toHaveBeenCalledTimes(1);
    const [nextState, prevState] = mapper.mock.calls[0] as [
      CountStore,
      CountStore,
    ];
    expect(prevState.count).toBe(0);
    expect(nextState.count).toBe(1);
  });

  it("does not block the synchronous Zustand state update", () => {
    type CountStore = { count: number; increment: () => void };
    const mapper: SemanticMapper<CountStore> = () => "User added item to cart";

    const useStore = create<CountStore>(
      semanticMiddleware(engine, mapper, (set) => ({
        count: 0,
        increment: () => set((s) => ({ count: s.count + 1 })),
      })),
    );

    // Calling increment() should immediately update count in the store.
    useStore.getState().increment();
    expect(useStore.getState().count).toBe(1);
  });
});
