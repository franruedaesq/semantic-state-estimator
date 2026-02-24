import { useSyncExternalStore } from "react";
import type { SemanticStateEngine } from "../engine/SemanticStateEngine.js";

/**
 * React hook that subscribes to a `SemanticStateEngine` and returns a
 * point-in-time snapshot of its state.
 *
 * Built on `useSyncExternalStore` so it is safe for concurrent React and
 * produces no tearing. The component re-renders automatically whenever the
 * engine fires a state-change notification via `subscribe`.
 */
export const useSemanticState = (engine: SemanticStateEngine) => {
  return useSyncExternalStore(
    (listener) => engine.subscribe(listener),
    () => engine.getSnapshot(),
  );
};
