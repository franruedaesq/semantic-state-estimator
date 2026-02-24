import type { StateCreator } from "zustand";
import type { SemanticStateEngine } from "../engine/SemanticStateEngine.js";

/**
 * Maps a Zustand state transition to a semantic string for the AI engine.
 * Return `null` to skip the update (lazy evaluation / noise filtering).
 */
export type SemanticMapper<T> = (
  state: T,
  previousState: T,
) => string | null | undefined;

/**
 * Zustand middleware that intercepts state changes and fires the semantic
 * engine asynchronously in the background (fire-and-forget).
 *
 * The `set` call is synchronous and unblocked; the engine runs in the
 * WebWorker so no frames are dropped.
 *
 * @param onError Optional callback invoked when `engine.update` rejects.
 *                Defaults to `console.error`.
 */
export const semanticMiddleware =
  <T>(
    engine: SemanticStateEngine,
    mapper: SemanticMapper<T>,
    config: StateCreator<T>,
    onError: (error: unknown) => void = console.error,
  ): StateCreator<T> =>
  (set, get, api) =>
    config(
      (partial, replace) => {
        const prevState = get();
        // Let Zustand perform the normal synchronous update first.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        set(partial as any, replace as any);
        const nextState = get();

        // Map the state change to a semantic string.
        const semanticText = mapper(nextState, prevState);

        // If valid, fire-and-forget to the WebWorker engine.
        if (semanticText) {
          engine.update(semanticText).catch(onError);
        }
      },
      get,
      api,
    );
