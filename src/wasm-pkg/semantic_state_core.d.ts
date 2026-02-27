/* tslint:disable */
/* eslint-disable */

/**
 * Core semantic-state engine, compiled to WebAssembly.
 *
 * Holds the EMA state vector and all associated metrics. Every call to
 * `update` performs vector math in Rust/WASM, returning an `UpdateResult`
 * that the TypeScript wrapper uses to fire the `onDriftDetected` callback.
 */
export class WasmStateEngine {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Returns a serialised `Snapshot` as a `JsValue`.
     */
    get_snapshot(now_ms: number): any;
    /**
     * Creates a new engine with the given EMA alpha and drift threshold.
     */
    constructor(alpha: number, drift_threshold: number);
    /**
     * Fuses a new embedding into the state.
     *
     * Accepts a `Float32Array` from JS, performs EMA fusion and drift
     * detection, then returns a serialised `UpdateResult` as a `JsValue`.
     *
     * # Errors
     * Returns an error string if the embedding is empty or its length
     * doesn't match the previously established dimension.
     */
    update(embedding: Float32Array, now_ms: number): any;
}

/**
 * Normalizes a `Float32Array` to unit length.
 * Exposed to JS so the TS wrapper can normalise embeddings before passing them.
 */
export function wasm_normalize(v: Float32Array): Float32Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmstateengine_free: (a: number, b: number) => void;
    readonly wasm_normalize: (a: number, b: number) => [number, number];
    readonly wasmstateengine_get_snapshot: (a: number, b: number) => [number, number, number];
    readonly wasmstateengine_new: (a: number, b: number) => number;
    readonly wasmstateengine_update: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
