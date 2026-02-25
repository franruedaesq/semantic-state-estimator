# semantic-state-estimator

[![npm version](https://img.shields.io/npm/v/semantic-state-estimator?style=flat-square)](https://www.npmjs.com/package/semantic-state-estimator)
[![license](https://img.shields.io/npm/l/semantic-state-estimator?style=flat-square)](LICENSE)
[![node](https://img.shields.io/node/v/semantic-state-estimator?style=flat-square)](https://nodejs.org)

**Bridge the gap between boolean UI state and semantic AI intent — all inside a WebWorker, on-device, zero-latency.**  
Instead of asking *"did the user click?"*, this library asks *"what does the user **mean**?"* — fusing local text embeddings with Exponential Moving Average (EMA) to build a living, drifting semantic context of your entire session.

## Installation

```bash
npm install semantic-state-estimator
```

> **Peer dependencies:** `react >=18` and/or `zustand >=4` are optional — only install what you need.

---

## Quick Start

### 1. Initialize the `SemanticStateEngine`

```typescript
import { WorkerManager, SemanticStateEngine } from 'semantic-state-estimator';

// The worker uses import.meta.url so Webpack 5 and Vite resolve the path correctly.
const workerManager = new WorkerManager();

const engine = new SemanticStateEngine({
  provider: workerManager,
  alpha: 0.5,           // Balanced EMA decay — see "Tuning the Math" below
  driftThreshold: 0.75, // Fire onDriftDetected when similarity drops below this
  onDriftDetected: (vector, driftScore) => {
    console.log(`Semantic drift detected! Score: ${driftScore.toFixed(3)}`);
  },
});

// Feed events into the engine — it runs inside the WebWorker, never blocks your UI
await engine.update('user opened the billing settings');
await engine.update('user clicked "cancel subscription"');

const snapshot = engine.getSnapshot();
console.log(snapshot.semanticSummary); // "stable" | "drifting" | "volatile"
```

### 2. Wrap a Zustand Store

```typescript
import { create } from 'zustand';
import { semanticMiddleware } from 'semantic-state-estimator/zustand';
import { WorkerManager, SemanticStateEngine } from 'semantic-state-estimator';

type AppState = {
  page: string;
  cartItems: number;
  setPage: (page: string) => void;
  addToCart: () => void;
};

const workerManager = new WorkerManager();
const engine = new SemanticStateEngine({ provider: workerManager, alpha: 0.5, driftThreshold: 0.75 });

// Wrap your store creator with semanticMiddleware
export const useAppStore = create<AppState>(
  semanticMiddleware(
    engine,
    // Map each state transition to a semantic string — return null to skip
    (next, prev) => {
      if (next.page !== prev.page) return `user navigated to ${next.page}`;
      if (next.cartItems > prev.cartItems) return 'user added item to cart';
      return null;
    },
    (set) => ({
      page: 'home',
      cartItems: 0,
      setPage: (page) => set({ page }),
      addToCart: () => set((s) => ({ cartItems: s.cartItems + 1 })),
    }),
  ),
);
```

---

## Tuning the Math: The EMA α (Decay) Weight

The `alpha` parameter controls how quickly new events override the session history.

| α value | Behavior | Best for |
|---------|-----------|----------|
| `0.1` | **Slow drift, highly stable.** Requires many consistent events to shift the state. Past context dominates. | Long-running sessions, background intent tracking |
| `0.5` | **Balanced.** Responds well to recent events while still remembering session history. | General-purpose apps, e-commerce, dashboards |
| `0.9` | **Highly reactive.** Almost instantly forgets past context in favour of the latest event. | Real-time chat, game UIs, live coding tools |

The EMA formula applied on every `engine.update(text)` call:

```
S_t = α · E_t + (1 − α) · S_{t−1}
```

Where `E_t` is the embedding of the incoming event and `S_{t−1}` is the previous state vector.

---

## The Drift Callback

The `onDriftDetected` callback fires **before** EMA fusion is applied, giving you a chance to react to a sharp semantic shift — e.g. a user suddenly switching from "browsing products" to "requesting a refund".

### With `SemanticStateEngine` directly

```typescript
const engine = new SemanticStateEngine({
  provider: workerManager,
  alpha: 0.5,
  driftThreshold: 0.75,
  onDriftDetected: (vector, driftScore) => {
    // driftScore = 1 − cosine_similarity ∈ [0, 2]
    if (driftScore > 0.8) {
      showModal('We noticed your focus shifted. Can we help?');
    }
  },
});
```

### With the React `useSemanticState` Hook

```tsx
import { useSemanticState } from 'semantic-state-estimator/react';
import { useEffect, useState } from 'react';

function SemanticStatusBanner({ engine }) {
  const [showDriftModal, setShowDriftModal] = useState(false);
  const snapshot = useSemanticState(engine); // re-renders on every engine.update()

  useEffect(() => {
    if (snapshot.semanticSummary === 'volatile') {
      setShowDriftModal(true);
    }
  }, [snapshot.semanticSummary]);

  return (
    <>
      <div>Health: {(snapshot.healthScore * 100).toFixed(0)}%</div>
      <div>State: {snapshot.semanticSummary}</div>
      {showDriftModal && (
        <Modal onClose={() => setShowDriftModal(false)}>
          Your session intent has shifted significantly. Need help?
        </Modal>
      )}
    </>
  );
}
```

---

## API Reference

### `new WorkerManager(workerUrl?, modelName?)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `workerUrl` | `string \| URL` | `new URL('./embedding.worker.js', import.meta.url)` | Location of the compiled worker file |
| `modelName` | `string` | `"Xenova/all-MiniLM-L6-v2"` | HuggingFace model for text embeddings |

### `new SemanticStateEngine(config)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `provider` | `EmbeddingProvider` | *(required)* | Provides async embedding vectors. `WorkerManager` satisfies this interface out of the box; you can also pass a custom OpenAI, Ollama, or any other wrapper. |
| `alpha` | `number` | — | EMA decay factor α ∈ (0, 1] |
| `driftThreshold` | `number` | — | Cosine similarity below which drift fires |
| `onDriftDetected` | `(vector, driftScore) => void` | `undefined` | Callback on semantic drift |
| `modelName` | `string` | `"Xenova/all-MiniLM-L6-v2"` | Model name (informational) |

### `engine.getSnapshot()` → `Snapshot`

```typescript
{
  vector: number[];         // Current EMA state vector
  healthScore: number;      // Reliability [0, 1] — degrades with age and drift
  timestamp: number;        // Unix ms of last update
  semanticSummary: string;  // "stable" | "drifting" | "volatile"
}
```

---

## Custom Embedding Providers

The `SemanticStateEngine` accepts any object that implements the `EmbeddingProvider` interface:

```typescript
import type { EmbeddingProvider } from 'semantic-state-estimator';

interface EmbeddingProvider {
  getEmbedding(text: string): Promise<Float32Array | number[]>;
}
```

`WorkerManager` satisfies this interface automatically, so existing code continues to work. You can also write a thin wrapper to use any other embedding source:

### OpenAI Provider

```typescript
import type { EmbeddingProvider } from 'semantic-state-estimator';

class OpenAIProvider implements EmbeddingProvider {
  constructor(private apiKey: string, private model = "text-embedding-3-small") {}

  async getEmbedding(text: string): Promise<number[]> {
    const res = await fetch("https://api.openai.com/v1/embeddings", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({ input: text, model: this.model })
    });
    const data = await res.json();
    return data.data[0].embedding; // 1536-dimension array
  }
}

const engine = new SemanticStateEngine({
  alpha: 0.5,
  driftThreshold: 0.75,
  provider: new OpenAIProvider("sk-..."),
});
```

### Ollama Provider

```typescript
import type { EmbeddingProvider } from 'semantic-state-estimator';

class OllamaProvider implements EmbeddingProvider {
  constructor(private model = "nomic-embed-text", private url = "http://localhost:11434") {}

  async getEmbedding(text: string): Promise<number[]> {
    const res = await fetch(`${this.url}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: this.model, prompt: text })
    });
    const data = await res.json();
    return data.embedding; // 768-dimension array
  }
}

const engine = new SemanticStateEngine({
  alpha: 0.5,
  driftThreshold: 0.75,
  provider: new OllamaProvider(),
});
```

> ⚠️ **Frontend / high-frequency usage warning:** The built-in `WorkerManager` runs inference locally in the browser in ~20–50 ms. If you replace it with a remote provider such as `OpenAIProvider`, every `engine.update()` call incurs a 300 ms–800 ms network round-trip. When used with `semanticMiddleware` on rapid UI state changes, requests will queue up and you may hit API rate limits quickly. Remote providers are best suited for server-side or local-desktop applications where update frequency is low.

---

## License

MIT
