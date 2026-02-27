# Proposal: Rewrite Core Logic in Rust (WASM)

## Introduction

This document proposes a strategy to rewrite the core logic of `semantic-state-estimator` in Rust, compiling it to WebAssembly (WASM). By moving the computationally intensive vector operations and state management to Rust, we can achieve better performance, stricter type safety, and a more robust codebase while maintaining the existing TypeScript API surface for users.

## Goals

1.  **Performance**: Leverage Rust's zero-cost abstractions and SIMD capabilities (via WASM) for vector math operations (cosine similarity, normalization, EMA fusion).
2.  **Safety**: Utilize Rust's ownership model to prevent state management bugs.
3.  **Portability**: Create a portable WASM module that can run in browsers (Web Workers), Node.js, and other environments.
4.  **Zero-Refactor for Users**: The public API (`SemanticStateEngine`, `WorkerManager`, types) must remain **identical**. Users should not need to change their code.

## Architecture

The architecture will consist of two layers:

1.  **Rust Core (`semantic-state-core`)**:
    -   A Rust crate compiled to WASM using `wasm-pack`.
    -   Contains the `SemanticState` struct which holds the state vector, drift metrics, and health score.
    -   Implements all vector math functions: `cosine_similarity`, `normalize`, `ema_fusion`.
    -   Exposes a class `WasmStateEngine` to JavaScript via `wasm-bindgen`.

2.  **TypeScript Wrapper (Existing `src` structure)**:
    -   The existing `SemanticStateEngine` class in `src/engine/SemanticStateEngine.ts` will be preserved but refactored to act as a wrapper around the `WasmStateEngine`.
    -   It will handle the asynchronous loading of the WASM module.
    -   It will continue to manage the `EmbeddingProvider` (which remains in JS/TS as it often involves network calls or other JS-based libraries like `@huggingface/transformers`).

### Diagram

```
[ User Application ]
       |
       v
[ SemanticStateEngine (TS Wrapper) ]  <-- Public API (unchanged)
       |
       +--> [ EmbeddingProvider (JS/TS) ] --> (Returns number[])
       |
       v
[ WasmStateEngine (Rust/WASM) ]       <-- Core Logic
       |
       +--> Stores State Vector (Vec<f32>)
       +--> Calculates EMA, Drift, Health
       +--> Returns Snapshot
```

## Implementation Plan

### Step 1: Rust Crate Setup

Create a new directory `crate/` (or `rust/`) in the project root. Initialize a generic library crate.

```bash
cargo new --lib crate
```

Add dependencies to `crate/Cargo.toml`:
- `wasm-bindgen`: For JS interop.
- `js-sys`: For interacting with JS objects (if needed).
- `getrandom`: For UUID generation (if needed, with `js` feature).

### Step 2: Porting Vector Math

Move the logic from `src/math/vector.ts` to Rust. Implement functions for:
- Dot product
- Magnitude (Norm)
- Cosine Similarity
- Scalar multiplication
- Vector addition
- Normalization

*Note: We can use the `nalgebra` crate or write pure Rust implementations for these simple operations to keep bundle size small.*

### Step 3: Porting `SemanticStateEngine` Logic

Create a struct `WasmStateEngine` in Rust that holds:
- `state_vector`: `Vec<f32>`
- `alpha`: `f32`
- `drift_threshold`: `f32`
- `last_updated_at`: `f64` (timestamp)
- `last_drift`: `f32`

Implement the `update` method:
1.  Accept a new embedding vector (`Vec<f32>` or `Float32Array`).
2.  Calculate cosine similarity.
3.  Detect drift.
4.  Update state using EMA.
5.  Update timestamps and health metrics.

### Step 4: Build & Bundling

We need to ensure the WASM file is easily consumable without complex bundler configuration for the user.

1.  Use `wasm-pack build --target web` to generate the WASM and JS glue code.
2.  **Inline WASM**: To avoid issues with loading `.wasm` files (MIME types, paths), we can base64-encode the `.wasm` file and inline it into the JS wrapper. This increases bundle size slightly (approx 33%) but guarantees portability across all environments (Webpack, Vite, Node, etc.).

### Step 5: TypeScript Integration

Modify `src/engine/SemanticStateEngine.ts`:

1.  **Async Initialization**: The WASM module must be loaded asynchronously.
    -   We might need a static `create` method or an `init()` method on the class.
    -   *Alternative*: Keep the constructor synchronous but queue operations until WASM is ready.

2.  **Refactor `update` method**:
    -   Call `provider.getEmbedding()` (JS).
    -   Pass the result to `wasmEngine.update(embedding)` (Rust).
    -   If drift is detected (returned by Rust or via callback), invoke the JS `onDriftDetected` callback.

## Backward Compatibility Strategy

To ensure users don't need to refactor:

1.  **Keep Types**: `Snapshot`, `SemanticStateEngineConfig` interfaces remain in `src/engine/SemanticStateEngine.ts`.
2.  **Keep API**: The `update(text: string)` and `getSnapshot()` methods will have the exact same signature.
3.  **Internal Change Only**: The switch to WASM will be an implementation detail hidden behind the class.

### Handling `onDriftDetected`
The Rust `update` method can return a struct indicating if drift occurred:

```rust
pub struct UpdateResult {
    pub drift_detected: bool,
    pub drift_score: f32,
    pub vector: Vec<f32>, // The input vector, needed for the callback
}
```

The TS wrapper receives this and calls the user's callback:

```typescript
// Inside SemanticStateEngine.ts
const result = this.wasmEngine.update(embedding);
if (result.drift_detected) {
    this.onDriftDetected?.(result.vector, result.drift_score);
}
```

## Benefits Summary

-   **Robustness**: Core logic is verified by the Rust compiler.
-   **Speed**: Heavy math operations run in WASM.
-   **Future-Proofing**: Easier to extend with more complex math/models in Rust.
-   **User Experience**: Unchanged. Drop-in replacement.
