/**
 * semantic-state-estimator
 *
 * A TypeScript library that acts as an event-stream middleware to track
 * the implicit semantic intent, emotional state, or "vibe" of a user/system.
 */

export { emaFusion, cosineSimilarity, normalize } from "./vectorMath.js";
export { add, scale } from "./math/vector.js";
export {
  SemanticStateEngine,
  type SemanticStateEngineOptions,
  type SemanticStateSnapshot,
} from "./SemanticStateEngine.js";
export { WorkerManager } from "./worker/WorkerManager.js";
