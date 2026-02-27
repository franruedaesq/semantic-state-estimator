use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

// ── Vector Math ───────────────────────────────────────────────────────────────

/// Computes the dot product of two equal-length slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Computes the L2 magnitude (norm) of a vector.
fn magnitude(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Computes the cosine similarity between two equal-length vectors.
/// Returns 0.0 if either vector has zero magnitude.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mag_a = magnitude(a);
    let mag_b = magnitude(b);
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    (dot(a, b) / (mag_a * mag_b)).clamp(-1.0, 1.0)
}

/// Normalizes a vector to unit length (L2). Returns a zero vector if magnitude is 0.
fn normalize(v: &[f32]) -> Vec<f32> {
    let mag = magnitude(v);
    if mag == 0.0 {
        return vec![0.0; v.len()];
    }
    v.iter().map(|x| x / mag).collect()
}

/// EMA fusion: S_t = α · E_t + (1 − α) · S_{t-1}
fn ema_fusion(current: &[f32], previous: &[f32], alpha: f32) -> Vec<f32> {
    current
        .iter()
        .zip(previous.iter())
        .map(|(c, p)| alpha * c + (1.0 - alpha) * p)
        .collect()
}

// ── UpdateResult ─────────────────────────────────────────────────────────────

/// The result returned from `WasmStateEngine::update`.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateResult {
    /// Whether drift was detected on this update.
    pub drift_detected: bool,
    /// Drift magnitude: 1 − cosine_similarity ∈ [0, 2].
    pub drift_score: f32,
    /// The input embedding (used by the TS wrapper to pass to `onDriftDetected`).
    pub vector: Vec<f32>,
}

// ── Snapshot ─────────────────────────────────────────────────────────────────

/// A point-in-time snapshot of the semantic state.
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    /// Current EMA state vector.
    pub vector: Vec<f32>,
    /// Health score in [0, 1].
    pub health_score: f32,
    /// Unix timestamp (ms) of the last update.
    pub timestamp: f64,
    /// Human-readable quality description.
    pub semantic_summary: String,
}

// ── WasmStateEngine ───────────────────────────────────────────────────────────

const AGE_DECAY_RATE: f32 = 0.0001;
const DRIFT_WEIGHT: f32 = 0.5;

/// Core semantic-state engine, compiled to WebAssembly.
///
/// Holds the EMA state vector and all associated metrics. Every call to
/// `update` performs vector math in Rust/WASM, returning an `UpdateResult`
/// that the TypeScript wrapper uses to fire the `onDriftDetected` callback.
#[wasm_bindgen]
pub struct WasmStateEngine {
    alpha: f32,
    drift_threshold: f32,
    state_vector: Vec<f32>,
    last_updated_at: f64,
    last_drift: f32,
    update_count: u32,
}

#[wasm_bindgen]
impl WasmStateEngine {
    /// Creates a new engine with the given EMA alpha and drift threshold.
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f32, drift_threshold: f32) -> WasmStateEngine {
        WasmStateEngine {
            alpha,
            drift_threshold,
            state_vector: Vec::new(),
            last_updated_at: 0.0,
            last_drift: 0.0,
            update_count: 0,
        }
    }

    /// Fuses a new embedding into the state.
    ///
    /// Accepts a `Float32Array` from JS, performs EMA fusion and drift
    /// detection, then returns a serialised `UpdateResult` as a `JsValue`.
    ///
    /// # Errors
    /// Returns an error string if the embedding is empty or its length
    /// doesn't match the previously established dimension.
    #[wasm_bindgen]
    pub fn update(&mut self, embedding: &[f32], now_ms: f64) -> Result<JsValue, JsValue> {
        if embedding.is_empty() {
            return Err(JsValue::from_str("Embedding must not be empty"));
        }

        let (drift_detected, drift_score) = if self.update_count == 0 {
            // First call: establish baseline from a zero-vector origin.
            let zero = vec![0.0f32; embedding.len()];
            self.state_vector = ema_fusion(embedding, &zero, self.alpha);
            (false, 0.0f32)
        } else {
            if embedding.len() != self.state_vector.len() {
                return Err(JsValue::from_str(&format!(
                    "Embedding dimension mismatch: expected {}, got {}",
                    self.state_vector.len(),
                    embedding.len()
                )));
            }

            let similarity = cosine_similarity(&self.state_vector, embedding);
            let drift = 1.0 - similarity;
            let detected = similarity < self.drift_threshold;

            self.state_vector = ema_fusion(embedding, &self.state_vector, self.alpha);
            self.last_drift = drift;
            (detected, drift)
        };

        self.last_updated_at = now_ms;
        self.update_count += 1;

        let result = UpdateResult {
            drift_detected,
            drift_score,
            vector: embedding.to_vec(),
        };

        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Returns a serialised `Snapshot` as a `JsValue`.
    #[wasm_bindgen]
    pub fn get_snapshot(&self, now_ms: f64) -> Result<JsValue, JsValue> {
        let health_score = self.calculate_health(now_ms);
        let snapshot = Snapshot {
            vector: self.state_vector.clone(),
            health_score,
            timestamp: self.last_updated_at,
            semantic_summary: Self::build_summary(health_score),
        };
        serde_wasm_bindgen::to_value(&snapshot).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn calculate_health(&self, now_ms: f64) -> f32 {
        let time_since_update = (now_ms - self.last_updated_at).max(0.0) as f32;
        let age_penalty = time_since_update * AGE_DECAY_RATE;
        let drift_penalty = self.last_drift * DRIFT_WEIGHT;
        (1.0 - age_penalty - drift_penalty).clamp(0.0, 1.0)
    }

    fn build_summary(health_score: f32) -> String {
        if health_score > 0.8 {
            "stable".to_string()
        } else if health_score > 0.5 {
            "drifting".to_string()
        } else {
            "volatile".to_string()
        }
    }
}

// ── Exported normalize (for completeness) ─────────────────────────────────────

/// Normalizes a `Float32Array` to unit length.
/// Exposed to JS so the TS wrapper can normalise embeddings before passing them.
#[wasm_bindgen]
pub fn wasm_normalize(v: &[f32]) -> Vec<f32> {
    normalize(v)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn test_dot_product() {
        assert!(approx_eq(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0));
    }

    #[test]
    fn test_magnitude() {
        assert!(approx_eq(magnitude(&[3.0, 4.0]), 5.0));
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        assert!(approx_eq(cosine_similarity(&v, &v), 1.0));
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        assert!(approx_eq(
            cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]),
            0.0
        ));
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        assert!(approx_eq(
            cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]),
            -1.0
        ));
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        assert!(approx_eq(
            cosine_similarity(&[0.0, 0.0], &[1.0, 2.0]),
            0.0
        ));
    }

    #[test]
    fn test_normalize() {
        let result = normalize(&[3.0, 4.0]);
        assert!(approx_eq(result[0], 0.6));
        assert!(approx_eq(result[1], 0.8));
    }

    #[test]
    fn test_normalize_zero_vector() {
        let result = normalize(&[0.0, 0.0, 0.0]);
        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_ema_fusion() {
        let result = ema_fusion(&[1.0, 0.0], &[0.0, 1.0], 0.5);
        assert!(approx_eq(result[0], 0.5));
        assert!(approx_eq(result[1], 0.5));
    }

    #[test]
    fn test_ema_fusion_alpha_one() {
        let result = ema_fusion(&[3.0, 1.0, 4.0], &[0.0, 0.0, 0.0], 1.0);
        assert_eq!(result, vec![3.0, 1.0, 4.0]);
    }
}
