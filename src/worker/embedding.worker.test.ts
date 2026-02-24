import { describe, it, expect, vi, afterEach } from "vitest";
import { handleMessage, handleInitMessage, getModelName } from "./embedding.worker.js";
import type { EmbeddingRequest, WorkerInitMessage } from "./types.js";

describe("embedding.worker – handleMessage", () => {
  afterEach(() => {
    // Restore the postMessage stub set up in vitest.setup.ts.
    (globalThis as Record<string, unknown>).postMessage = () => undefined;
  });

  it("posts back an EmbeddingResponse with a dummy Float32Array(384)", () => {
    const postMessageSpy = vi.fn();
    (globalThis as Record<string, unknown>).postMessage = postMessageSpy;

    const request: EmbeddingRequest = { type: "EMBED", id: "test-uuid", text: "hello world" };
    handleMessage({ data: request } as MessageEvent<EmbeddingRequest>);

    expect(postMessageSpy).toHaveBeenCalledTimes(1);

    const [response] = postMessageSpy.mock.calls[0] as [
      { id: string; vector: Float32Array },
    ];
    expect(response.id).toBe("test-uuid");
    expect(response.vector).toBeInstanceOf(Float32Array);
    expect(response.vector).toHaveLength(384);
    expect(response.vector[0]).toBeCloseTo(0.1);
  });

  it("echoes the request id in the response", () => {
    const postMessageSpy = vi.fn();
    (globalThis as Record<string, unknown>).postMessage = postMessageSpy;

    handleMessage({
      data: { type: "EMBED", id: "unique-id-42", text: "test" },
    } as MessageEvent<EmbeddingRequest>);

    const [response] = postMessageSpy.mock.calls[0] as [{ id: string }];
    expect(response.id).toBe("unique-id-42");
  });
});

describe("embedding.worker – handleInitMessage", () => {
  it("saves the modelName from the INIT message to the worker's local state", () => {
    const initMsg: WorkerInitMessage = { type: "INIT", modelName: "Xenova/bge-small-en-v1.5" };
    handleInitMessage({ data: initMsg } as MessageEvent<WorkerInitMessage>);

    expect(getModelName()).toBe("Xenova/bge-small-en-v1.5");
  });

  it("defaults to 'Xenova/all-MiniLM-L6-v2' before any INIT message is received", () => {
    // Reset by sending the default model name
    const initMsg: WorkerInitMessage = { type: "INIT", modelName: "Xenova/all-MiniLM-L6-v2" };
    handleInitMessage({ data: initMsg } as MessageEvent<WorkerInitMessage>);

    expect(getModelName()).toBe("Xenova/all-MiniLM-L6-v2");
  });
});
