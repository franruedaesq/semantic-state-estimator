import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { WorkerManager } from "./WorkerManager.js";
import type { EmbeddingRequest, EmbeddingResponse, WorkerIncomingMessage, WorkerStatusEvent } from "./types.js";

// ─── Mock Worker helpers ──────────────────────────────────────────────────────

type MockWorkerInstance = {
  onmessage: ((event: { data: EmbeddingResponse | WorkerStatusEvent }) => void) | null;
  postMessage: ReturnType<typeof vi.fn>;
  addEventListener: ReturnType<typeof vi.fn>;
  terminate: ReturnType<typeof vi.fn>;
};

/**
 * Creates a mock Worker class that:
 * 1. Immediately fires a STATUS ready event on INIT (so isReady becomes true).
 * 2. Resolves EMBED requests with `responseVector`.
 */
function makeResolvingWorkerClass(
  responseVector: Float32Array,
): ReturnType<typeof vi.fn> {
  return vi.fn().mockImplementation(() => {
    const inst: MockWorkerInstance = {
      onmessage: null,
      postMessage: vi.fn().mockImplementation((req: WorkerIncomingMessage) => {
        if (req.type === "INIT") {
          setTimeout(() => {
            inst.onmessage?.({ data: { type: 'STATUS', status: 'ready' } satisfies WorkerStatusEvent });
          }, 0);
          return;
        }
        if (req.type !== "EMBED") return;
        setTimeout(() => {
          inst.onmessage?.({
            data: { type: 'EMBED_RES', id: req.id, vector: responseVector } satisfies EmbeddingResponse,
          });
        }, 0);
      }),
      addEventListener: vi.fn(),
      terminate: vi.fn(),
    };
    return inst;
  });
}

/**
 * Creates a mock Worker class that:
 * 1. Immediately fires a STATUS ready event on INIT (so isReady becomes true).
 * 2. Rejects EMBED requests with `errorMessage`.
 */
function makeRejectingWorkerClass(
  errorMessage: string,
): ReturnType<typeof vi.fn> {
  return vi.fn().mockImplementation(() => {
    const inst: MockWorkerInstance = {
      onmessage: null,
      postMessage: vi.fn().mockImplementation((req: WorkerIncomingMessage) => {
        if (req.type === "INIT") {
          setTimeout(() => {
            inst.onmessage?.({ data: { type: 'STATUS', status: 'ready' } satisfies WorkerStatusEvent });
          }, 0);
          return;
        }
        if (req.type !== "EMBED") return;
        setTimeout(() => {
          inst.onmessage?.({
            data: {
              type: 'EMBED_RES',
              id: req.id,
              vector: null,
              error: errorMessage,
            } satisfies EmbeddingResponse,
          });
        }, 0);
      }),
      addEventListener: vi.fn(),
      terminate: vi.fn(),
    };
    return inst;
  });
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("WorkerManager", () => {
  let savedWorker: unknown;

  beforeEach(() => {
    savedWorker = (globalThis as Record<string, unknown>).Worker;
  });

  afterEach(() => {
    (globalThis as Record<string, unknown>).Worker = savedWorker;
  });

  it("instantiates a Worker on construction", () => {
    const MockWorkerClass = vi.fn().mockImplementation(() => ({
      onmessage: null,
      postMessage: vi.fn(),
      addEventListener: vi.fn(),
      terminate: vi.fn(),
    }));
    (globalThis as Record<string, unknown>).Worker = MockWorkerClass;

    new WorkerManager("embedding.worker.js");

    expect(MockWorkerClass).toHaveBeenCalledTimes(1);
  });

  it("immediately posts an INIT message with the default modelName on construction", () => {
    let workerInstance: MockWorkerInstance | null = null;
    const MockWorkerClass = vi.fn().mockImplementation(() => {
      workerInstance = {
        onmessage: null,
        postMessage: vi.fn(),
        addEventListener: vi.fn(),
        terminate: vi.fn(),
      };
      return workerInstance;
    });
    (globalThis as Record<string, unknown>).Worker = MockWorkerClass;

    new WorkerManager("embedding.worker.js");

    expect(workerInstance!.postMessage).toHaveBeenCalledTimes(1);
    expect(workerInstance!.postMessage).toHaveBeenCalledWith({
      type: "INIT",
      modelName: "Xenova/all-MiniLM-L6-v2",
    });
  });

  it("posts an INIT message with the provided modelName on construction", () => {
    let workerInstance: MockWorkerInstance | null = null;
    const MockWorkerClass = vi.fn().mockImplementation(() => {
      workerInstance = {
        onmessage: null,
        postMessage: vi.fn(),
        addEventListener: vi.fn(),
        terminate: vi.fn(),
      };
      return workerInstance;
    });
    (globalThis as Record<string, unknown>).Worker = MockWorkerClass;

    new WorkerManager("embedding.worker.js", "Xenova/bge-small-en-v1.5");

    expect(workerInstance!.postMessage).toHaveBeenCalledWith({
      type: "INIT",
      modelName: "Xenova/bge-small-en-v1.5",
    });
  });

  it("resolves getEmbedding with the Float32Array returned by the worker", async () => {
    const vector = new Float32Array(384).fill(0.1);
    (globalThis as Record<string, unknown>).Worker =
      makeResolvingWorkerClass(vector);

    const manager = new WorkerManager("embedding.worker.js");
    // Wait for STATUS ready event to be processed
    await new Promise((r) => setTimeout(r, 10));

    const result = await manager.getEmbedding("user clicked cancel");

    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toHaveLength(384);
    expect((result as Float32Array)[0]).toBeCloseTo(0.1);
  });

  it("rejects getEmbedding when the worker replies with an error", async () => {
    (globalThis as Record<string, unknown>).Worker =
      makeRejectingWorkerClass("Model failed to load");

    const manager = new WorkerManager("embedding.worker.js");
    // Wait for STATUS ready event to be processed
    await new Promise((r) => setTimeout(r, 10));

    await expect(manager.getEmbedding("hello")).rejects.toThrow(
      "Model failed to load",
    );
  });

  it("resolves getEmbedding with null when the worker is not yet ready", async () => {
    const MockWorkerClass = vi.fn().mockImplementation(() => ({
      onmessage: null,
      postMessage: vi.fn(),
      addEventListener: vi.fn(),
      terminate: vi.fn(),
    }));
    (globalThis as Record<string, unknown>).Worker = MockWorkerClass;

    const manager = new WorkerManager("embedding.worker.js");
    // Do NOT wait – worker has not sent STATUS ready yet

    await expect(manager.getEmbedding("early call")).resolves.toBeNull();
  });

  it("becomes ready after receiving STATUS ready from the worker", async () => {
    const vector = new Float32Array(384).fill(0.2);
    (globalThis as Record<string, unknown>).Worker =
      makeResolvingWorkerClass(vector);

    const manager = new WorkerManager("embedding.worker.js");
    // Before STATUS ready arrives, requests should resolve with null
    await expect(manager.getEmbedding("too early")).resolves.toBeNull();

    // Wait for STATUS ready
    await new Promise((r) => setTimeout(r, 10));

    // Now requests should succeed
    const result = await manager.getEmbedding("after ready");
    expect(result).toBeInstanceOf(Float32Array);
  });

  it("creates a Blob URL worker when no workerUrl is provided", () => {
    // The workerCode stub exports an empty string, so the Blob will contain "".
    // We only care that createObjectURL was called with a Blob.
    const fakeUrl = "blob:http://localhost/fake-worker-id";
    const createObjectURLSpy = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue(fakeUrl);

    const MockWorkerClass = vi.fn().mockImplementation(() => ({
      onmessage: null,
      onerror: null,
      postMessage: vi.fn(),
      addEventListener: vi.fn(),
      terminate: vi.fn(),
    }));
    (globalThis as Record<string, unknown>).Worker = MockWorkerClass;

    new WorkerManager(); // no URL argument

    expect(createObjectURLSpy).toHaveBeenCalledTimes(1);
    expect(createObjectURLSpy).toHaveBeenCalledWith(expect.any(Blob));
    expect(MockWorkerClass).toHaveBeenCalledWith(fakeUrl, { type: "classic" });

    createObjectURLSpy.mockRestore();
  });

  it("instantiates Worker with { type: 'classic' } when an explicit URL is provided", () => {
    // Verifies that the explicit-workerUrl code path also uses 'classic'.
    // CJS-bundled inlined code has no `require()` when executed inside a module
    // worker, so 'classic' is required for both Blob and explicit-URL paths.
    const MockWorkerClass = vi.fn().mockImplementation(() => ({
      onmessage: null,
      onerror: null,
      postMessage: vi.fn(),
      addEventListener: vi.fn(),
      terminate: vi.fn(),
    }));
    (globalThis as Record<string, unknown>).Worker = MockWorkerClass;

    new WorkerManager("embedding.worker.js");

    expect(MockWorkerClass).toHaveBeenCalledWith(
      "embedding.worker.js",
      { type: "classic" },
    );
  });

  it("wires onerror and logs to console.error on worker error", () => {
    let workerInstance: {
      onmessage: ((e: { data: EmbeddingResponse | WorkerStatusEvent }) => void) | null;
      onerror: ((e: ErrorEvent) => void) | null;
      postMessage: ReturnType<typeof vi.fn>;
      addEventListener: ReturnType<typeof vi.fn>;
      terminate: ReturnType<typeof vi.fn>;
    } | null = null;

    const MockWorkerClass = vi.fn().mockImplementation(() => {
      workerInstance = {
        onmessage: null,
        onerror: null,
        postMessage: vi.fn(),
        addEventListener: vi.fn(),
        terminate: vi.fn(),
      };
      return workerInstance;
    });
    (globalThis as Record<string, unknown>).Worker = MockWorkerClass;

    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => { });

    new WorkerManager("embedding.worker.js");

    // Simulate a worker error (ErrorEvent is not available in jsdom/Node)
    const fakeErrorEvent = { type: "error", message: "script failed" } as unknown as ErrorEvent;
    workerInstance!.onerror!(fakeErrorEvent);

    expect(consoleSpy).toHaveBeenCalledTimes(1);
    expect(consoleSpy.mock.calls[0][0]).toContain("fatal error");

    consoleSpy.mockRestore();
  });
})
