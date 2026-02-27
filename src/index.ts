/**
 * semantic-state-estimator
 *
 * A TypeScript library that acts as an event-stream middleware to track
 * the implicit semantic intent, emotional state, or "vibe" of a user/system.
 */

export { emaFusion, cosineSimilarity, normalize, add, scale } from "./math/vector.js";
export {
  SemanticStateEngine,
  type EmbeddingProvider,
  type SemanticStateEngineConfig,
  type Snapshot,
} from "./engine/SemanticStateEngine.js";
export { WorkerManager } from "./worker/WorkerManager.js";
export {
  SemanticStateError,
  DimensionMismatchError,
  EmbeddingProviderError,
} from "./errors.js";
