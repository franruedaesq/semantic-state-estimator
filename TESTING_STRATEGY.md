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

To guarantee that users can upgrade without any code changes, we employ a multi-layered testing strategy:

### 1. Regression Testing (Existing Suite)
We run the existing Vitest suite (`npm test`) to ensure that all public contracts remain satisfied. This includes:
*   `SemanticStateEngine` API methods (`update`, `getSnapshot`, `subscribe`).
*   `WorkerManager` communication and error handling.
*   React hooks and Zustand middleware integration.

### 2. Shadow / Parity Testing (New)
This is the primary mechanism for verifying behavioral correctness. We implement a **Shadow Engine**—a pure TypeScript reimplementation of the core logic (effectively the "old" version logic)—and run it side-by-side with the new WASM engine.

**The "Golden Master" Approach:**
*   **Setup**: Instantiate both `SemanticStateEngine` (WASM) and `ShadowStateEngine` (TS) with identical configuration (`alpha`, `driftThreshold`).
*   **Execution**: Feed identical random embedding vectors (Fuzzing) to both engines in a loop.
*   **Verification**: After *every* update, compare the internal state:
    *   **State Vector**: Must match element-wise (within floating-point tolerance).
    *   **Health Score**: Must be identical.
    *   **Drift Detection**: Both must fire `onDriftDetected` at the same step with the same score.
    *   **Semantic Summary**: Must be identical.

### 3. Fuzz Testing
We generate random vectors (unit length, zero vectors, orthogonal vectors) to test edge cases and ensure the WASM implementation handles numerical stability (NaNs, Infinity) as gracefully as the TypeScript version.

## Goal

The ultimate goal is **Behavioral Parity**. A user upgrading to this version should observe no difference in the semantic state values, drift sensitivity, or health metrics, ensuring a seamless drop-in replacement.
