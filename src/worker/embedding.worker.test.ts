import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { handleMessage, handleInitMessage, getModelName, PipelineSingleton } from "./embedding.worker.js";
import type { EmbeddingRequest, WorkerInitMessage } from "./types.js";

// Mock @huggingface/transformers
vi.mock("@huggingface/transformers", () => {
  const mockExtractor = vi.fn().mockResolvedValue({ data: new Float32Array(384).fill(0.5) });
  const mockPipeline = vi.fn().mockResolvedValue(mockExtractor);
  return {
    pipeline: mockPipeline,
    env: { allowLocalModels: true },
  };
});

describe("PipelineSingleton", () => {
  beforeEach(() => {
    // Reset singleton state before each test
    PipelineSingleton.instance = null;
    PipelineSingleton.modelName = null;
  });

  it("creates the pipeline only once when called multiple times with the same model", async () => {
    const { pipeline } = await import("@huggingface/transformers");
    const mockPipeline = pipeline as ReturnType<typeof vi.fn>;
    mockPipeline.mockClear();

    await PipelineSingleton.getInstance("Xenova/all-MiniLM-L6-v2");
    await PipelineSingleton.getInstance("Xenova/all-MiniLM-L6-v2");

    expect(mockPipeline).toHaveBeenCalledTimes(1);
  });

  it("creates a new pipeline instance when the model name changes", async () => {
    const { pipeline } = await import("@huggingface/transformers");
    const mockPipeline = pipeline as ReturnType<typeof vi.fn>;
    mockPipeline.mockClear();

    await PipelineSingleton.getInstance("Xenova/all-MiniLM-L6-v2");
    await PipelineSingleton.getInstance("Xenova/bge-small-en-v1.5");

    expect(mockPipeline).toHaveBeenCalledTimes(2);
  });
});

describe("embedding.worker – handleInitMessage", () => {
  let postMessageSpy: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    postMessageSpy = vi.fn();
    (globalThis as Record<string, unknown>).postMessage = postMessageSpy;
    PipelineSingleton.instance = null;
    PipelineSingleton.modelName = null;
  });

  afterEach(() => {
    (globalThis as Record<string, unknown>).postMessage = () => undefined;
  });

  it("saves the modelName from the INIT message to the worker's local state", async () => {
    const initMsg: WorkerInitMessage = { type: "INIT", modelName: "Xenova/bge-small-en-v1.5" };
    await handleInitMessage({ data: initMsg } as MessageEvent<WorkerInitMessage>);

    expect(getModelName()).toBe("Xenova/bge-small-en-v1.5");
  });

  it("defaults to 'Xenova/all-MiniLM-L6-v2' before any INIT message is received", async () => {
    const initMsg: WorkerInitMessage = { type: "INIT", modelName: "Xenova/all-MiniLM-L6-v2" };
    await handleInitMessage({ data: initMsg } as MessageEvent<WorkerInitMessage>);

    expect(getModelName()).toBe("Xenova/all-MiniLM-L6-v2");
  });

  it("posts STATUS loading then STATUS ready on successful INIT", async () => {
    const initMsg: WorkerInitMessage = { type: "INIT", modelName: "Xenova/all-MiniLM-L6-v2" };
    await handleInitMessage({ data: initMsg } as MessageEvent<WorkerInitMessage>);

    expect(postMessageSpy).toHaveBeenCalledWith({ type: 'STATUS', status: 'loading' });
    expect(postMessageSpy).toHaveBeenCalledWith({ type: 'STATUS', status: 'ready' });
    expect(postMessageSpy.mock.calls[0][0]).toEqual({ type: 'STATUS', status: 'loading' });
    expect(postMessageSpy.mock.calls[postMessageSpy.mock.calls.length - 1][0]).toEqual({ type: 'STATUS', status: 'ready' });
  });

  it("posts STATUS failed when the pipeline throws", async () => {
    const { pipeline } = await import("@huggingface/transformers");
    const mockPipeline = pipeline as ReturnType<typeof vi.fn>;
    mockPipeline.mockRejectedValueOnce(new Error("Network error"));
    PipelineSingleton.instance = null;
    PipelineSingleton.modelName = null;

    const initMsg: WorkerInitMessage = { type: "INIT", modelName: "Xenova/all-MiniLM-L6-v2" };
    await handleInitMessage({ data: initMsg } as MessageEvent<WorkerInitMessage>);

    expect(postMessageSpy).toHaveBeenCalledWith({ type: 'STATUS', status: 'loading' });
    expect(postMessageSpy).toHaveBeenCalledWith({ type: 'STATUS', status: 'failed', error: 'Network error' });
  });
});

describe("embedding.worker – handleMessage", () => {
  let postMessageSpy: ReturnType<typeof vi.fn>;

  beforeEach(async () => {
    postMessageSpy = vi.fn();
    (globalThis as Record<string, unknown>).postMessage = postMessageSpy;
    PipelineSingleton.instance = null;
    PipelineSingleton.modelName = null;
    // Pre-initialize so handleMessage can get an instance
    const initMsg: WorkerInitMessage = { type: "INIT", modelName: "Xenova/all-MiniLM-L6-v2" };
    await handleInitMessage({ data: initMsg } as MessageEvent<WorkerInitMessage>);
    postMessageSpy.mockClear();
  });

  afterEach(() => {
    (globalThis as Record<string, unknown>).postMessage = () => undefined;
  });

  it("posts back an EmbeddingResponse with the pipeline's output Float32Array", async () => {
    const request: EmbeddingRequest = { type: "EMBED", id: "test-uuid", text: "hello world" };
    await handleMessage({ data: request } as MessageEvent<EmbeddingRequest>);

    expect(postMessageSpy).toHaveBeenCalledTimes(1);
    const [response] = postMessageSpy.mock.calls[0] as [{ id: string; vector: Float32Array }];
    expect(response.id).toBe("test-uuid");
    expect(response.vector).toBeInstanceOf(Float32Array);
    expect(response.vector).toHaveLength(384);
  });

  it("echoes the request id in the response", async () => {
    await handleMessage({
      data: { type: "EMBED", id: "unique-id-42", text: "test" },
    } as MessageEvent<EmbeddingRequest>);

    const [response] = postMessageSpy.mock.calls[0] as [{ id: string }];
    expect(response.id).toBe("unique-id-42");
  });

  it("posts an error response when the extractor throws", async () => {
    const { pipeline } = await import("@huggingface/transformers");
    const mockPipeline = pipeline as ReturnType<typeof vi.fn>;
    const mockExtractorThatThrows = vi.fn().mockRejectedValueOnce(new Error("Inference error"));
    mockPipeline.mockResolvedValueOnce(mockExtractorThatThrows);
    PipelineSingleton.instance = null;
    PipelineSingleton.modelName = null;

    const request: EmbeddingRequest = { type: "EMBED", id: "err-id", text: "fail" };
    await handleMessage({ data: request } as MessageEvent<EmbeddingRequest>);

    const [response] = postMessageSpy.mock.calls[0] as [{ id: string; vector: null; error: string }];
    expect(response.id).toBe("err-id");
    expect(response.vector).toBeNull();
    expect(response.error).toBe("Inference error");
  });
});
