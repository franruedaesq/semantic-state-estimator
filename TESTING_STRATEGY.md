# Testing Strategy & Migration Verification

## Overview

This branch introduces a core architectural change: migrating the heavy computational logic (vector math, EMA fusion, state management) from TypeScript to **Rust (compiled to WebAssembly)**. This change aims to improve performance and type safety while maintaining **100% backward compatibility** with the existing TypeScript API.

## Core Features to Verify

The following core features have been moved to the Rust `WasmStateEngine`:

1.  **EMA Fusion**: Exponential Moving Average calculation `S_t = α · E_t + (1 − α) · S_{t−1}`.
2.  **Cosine Similarity & Drift Detection**: Detecting when a new embedding significantly diverges from the current state.
3.  **Health Score Calculation**: A composite score based on session age and accumulated drift.
4.  **Semantic Summary**: Categorizing state as "stable", "drifting", or "volatile".
5.  **WebWorker Offloading**: Ensuring the WASM engine runs correctly inside a WebWorker, preventing UI blocking.

## Testing Strategy

To guarantee that users can upgrade without any code changes, we employ a multi-layered testing strategy. The specific tests required are detailed below.

### 1. Regression Testing (Existing Suite)
We must run the existing Vitest suite (`npm test`) to ensure that all public contracts remain satisfied. This includes verifying:
*   `SemanticStateEngine` API methods (`update`, `getSnapshot`, `subscribe`) behave exactly as documented.
*   `WorkerManager` correctly handles async communication and error propagation.
*   React hooks and Zustand middleware integration points function without modification.

### 2. Shadow / Parity Testing (New - To Be Implemented)
This is the primary mechanism for verifying behavioral correctness. A **Shadow Engine**—a pure TypeScript reimplementation of the core logic (effectively the "old" version logic)—should be run side-by-side with the new WASM engine.

**Required Scenarios:**

*   **Initialization Parity**:
    *   Verify that both engines start with identical state vectors (typically zeros).
    *   Verify that both engines report an initial health score of 1.0.
    *   Verify that both engines produce identical initial snapshots.

*   **Steady State Evolution**:
    *   Feed a sequence of consistent (high similarity) embeddings to both engines.
    *   Verify that the state vector evolves identically after each update.
    *   Verify that the health score degrades identically due to time decay (mocking time is essential here).

*   **Drift Event Detection**:
    *   Inject a high-drift embedding (low cosine similarity).
    *   Verify that *both* engines fire `onDriftDetected` at the exact same step.
    *   Verify that the reported `driftScore` is identical (within floating-point tolerance).

*   **Recovery Phase**:
    *   After a drift event, return to consistent embeddings.
    *   Verify that the health score recovers at the same rate for both engines.

### 3. Fuzz Testing (New - To Be Implemented)
We must generate random vectors and edge-case inputs to ensure the WASM implementation handles numerical stability as gracefully as the TypeScript version.

**Specific Edge Cases:**
*   **Input Handling**: Test with empty strings, very long strings (token limit boundaries), and special characters (Unicode/Emoji).
*   **Vector Math**:
    *   **Zero Vectors**: Ensure division-by-zero protection in normalization/cosine similarity.
    *   **Orthogonal Vectors**: Ensure cosine similarity correctly reports 0.0.
    *   **Collinear Vectors**: Ensure cosine similarity correctly reports 1.0 (or -1.0).
    *   **Tiny/Huge Magnitudes**: Ensure no overflow/underflow issues in dot product calculations.

### 4. Performance Testing (Optional but Recommended)
*   **Execution Time**: Benchmark the `update()` method execution time for WASM vs TS implementation. Expectation: WASM should be faster or comparable.
*   **Memory Usage**: Monitor heap allocation during long-running sessions to ensure no memory leaks in the WASM linear memory.

### 5. Browser Compatibility
*   **WASM Loading**: Verify that the WASM module loads correctly in all target browsers (Chrome, Firefox, Safari, Edge).
*   **Fallback**: While not strictly required if we target modern browsers, verify behavior/error messages if WASM is not supported.

## Goal

The ultimate goal is **Behavioral Parity**. A user upgrading to this version should observe no difference in the semantic state values, drift sensitivity, or health metrics, ensuring a seamless drop-in replacement.
