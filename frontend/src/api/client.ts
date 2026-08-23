// Typed API client. Backend error envelopes are unwrapped into ApiError.

import type { Answer, IngestResult, Item, ItemDetail } from "../types";

// Default "/api" is proxied by the Vite dev server (same-origin, avoids CORS).
const BASE_URL = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly type: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError(
      "Cannot reach the server. Is the backend running?",
      0,
      "NetworkError"
    );
  }

  const body = await resp.json().catch(() => null);

  if (!resp.ok) {
    const err = body?.error;
    throw new ApiError(
      err?.message ?? `Request failed (${resp.status}).`,
      resp.status,
      err?.type ?? "UnknownError"
    );
  }
  return body as T;
}

export const api = {
  listItems: () =>
    request<{ items: Item[]; count: number }>("/items"),

  getItem: (id: string) => request<ItemDetail>(`/items/${id}`),

  updateItem: (id: string, body: { title?: string; content?: string }) =>
    request<ItemDetail>(`/items/${id}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),

  deleteItem: (id: string) =>
    request<void>(`/items/${id}`, { method: "DELETE" }),

  ingestNote: (text: string, title?: string) =>
    request<IngestResult>("/ingest", {
      method: "POST",
      body: JSON.stringify({ text, title: title || undefined }),
    }),

  ingestUrl: (url: string) =>
    request<IngestResult>("/ingest", {
      method: "POST",
      body: JSON.stringify({ url }),
    }),

  query: (question: string) =>
    request<Answer>("/query", {
      method: "POST",
      body: JSON.stringify({ question }),
    }),
};
