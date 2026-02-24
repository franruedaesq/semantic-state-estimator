import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { WorkerManager } from "./WorkerManager.js";
import type { EmbeddingRequest, EmbeddingResponse, WorkerIncomingMessage } from "./types.js";

// ─── Mock Worker helpers ──────────────────────────────────────────────────────

type MockWorkerInstance = {
  onmessage: ((event: { data: EmbeddingResponse }) => void) | null;
  postMessage: ReturnType<typeof vi.fn>;
  addEventListener: ReturnType<typeof vi.fn>;
  terminate: ReturnType<typeof vi.fn>;
};

/**
 * Creates a mock Worker class whose postMessage immediately simulates a
 * successful worker response carrying `responseVector` for EMBED requests.
 * INIT messages are silently ignored (no response needed).
 */
function makeResolvingWorkerClass(
  responseVector: Float32Array,
): ReturnType<typeof vi.fn> {
  return vi.fn().mockImplementation(() => {
    const inst: MockWorkerInstance = {
      onmessage: null,
      postMessage: vi.fn().mockImplementation((req: WorkerIncomingMessage) => {
        if (req.type !== "EMBED") return;
        setTimeout(() => {
          inst.onmessage?.({
            data: { id: req.id, vector: responseVector } satisfies EmbeddingResponse,
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
 * Creates a mock Worker class whose postMessage immediately simulates an error
 * response carrying `errorMessage` for EMBED requests.
 */
function makeRejectingWorkerClass(
  errorMessage: string,
): ReturnType<typeof vi.fn> {
  return vi.fn().mockImplementation(() => {
    const inst: MockWorkerInstance = {
      onmessage: null,
      postMessage: vi.fn().mockImplementation((req: WorkerIncomingMessage) => {
        if (req.type !== "EMBED") return;
        setTimeout(() => {
          inst.onmessage?.({
            data: {
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
    const result = await manager.getEmbedding("user clicked cancel");

    expect(result).toBeInstanceOf(Float32Array);
    expect(result).toHaveLength(384);
    expect(result[0]).toBeCloseTo(0.1);
  });

  it("rejects getEmbedding when the worker replies with an error", async () => {
    (globalThis as Record<string, unknown>).Worker =
      makeRejectingWorkerClass("Model failed to load");

    const manager = new WorkerManager("embedding.worker.js");
    await expect(manager.getEmbedding("hello")).rejects.toThrow(
      "Model failed to load",
    );
  });
});
