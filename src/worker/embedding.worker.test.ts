import { describe, it, expect, vi, afterEach } from "vitest";
import { handleMessage } from "./embedding.worker.js";
import type { EmbeddingRequest } from "./types.js";

describe("embedding.worker – handleMessage", () => {
  afterEach(() => {
    // Restore the postMessage stub set up in vitest.setup.ts.
    (globalThis as Record<string, unknown>).postMessage = () => undefined;
  });

  it("posts back an EmbeddingResponse with a dummy Float32Array(384)", () => {
    const postMessageSpy = vi.fn();
    (globalThis as Record<string, unknown>).postMessage = postMessageSpy;

    const request: EmbeddingRequest = { id: "test-uuid", text: "hello world" };
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
      data: { id: "unique-id-42", text: "test" },
    } as MessageEvent<EmbeddingRequest>);

    const [response] = postMessageSpy.mock.calls[0] as [{ id: string }];
    expect(response.id).toBe("unique-id-42");
  });
});
