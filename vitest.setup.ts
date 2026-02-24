/**
 * Vitest global setup for the semantic-state-estimator test suite.
 *
 * Adds the minimal browser globals required to test WebWorker-related code
 * inside the Node.js environment that Vitest uses.
 */

// Make `self` point to globalThis so worker code that calls `self.postMessage`
// or `self.addEventListener` resolves to the mocked globals below.
(globalThis as Record<string, unknown>).self = globalThis;

// Provide a no-op `postMessage` stub so worker modules can call it without
// throwing.  Individual tests override this with a spy as needed.
(globalThis as Record<string, unknown>).postMessage = () => undefined;

// Provide a no-op `addEventListener` stub so worker modules can call
// self.addEventListener('message', ...) without throwing.
// Individual tests can inspect or override this as needed.
(globalThis as Record<string, unknown>).addEventListener = () => undefined;

// Provide a minimal Worker class so WorkerManager (main-thread code) can call
// `new Worker(url)` in tests.  Individual tests replace this with a spy or a
// richer mock.
class Worker {
  onmessage: ((event: MessageEvent) => void) | null = null;

  constructor(public readonly url: string | URL) {}

  postMessage(_data: unknown): void {}

  addEventListener(
    _type: string,
    _handler: EventListenerOrEventListenerObject,
  ): void {}

  terminate(): void {}
}

(globalThis as Record<string, unknown>).Worker = Worker;
