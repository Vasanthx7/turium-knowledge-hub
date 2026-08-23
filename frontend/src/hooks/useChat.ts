// Conversation state for the RAG chat panel. Each ask appends a user turn then
// an assistant turn (answer with citations, or an error).

import { useCallback, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import type { ChatMessage } from "../types";

export interface UseChat {
  messages: ChatMessage[];
  asking: boolean;
  ask: (question: string) => Promise<void>;
  clear: () => void;
}

export function useChat(): UseChat {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [asking, setAsking] = useState(false);

  // Monotonic id source for stable React keys; avoids Date.now/random collisions.
  const seq = useRef(0);
  const nextId = () => `m${seq.current++}`;

  const append = useCallback((msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const ask = useCallback(
    async (question: string) => {
      if (asking) return;
      append({ id: nextId(), role: "user", text: question });
      setAsking(true);
      try {
        const answer = await api.query(question);
        append({
          id: nextId(),
          role: "assistant",
          text: answer.answer,
          citations: answer.citations,
        });
      } catch (e) {
        const message =
          e instanceof ApiError ? e.message : "Failed to get an answer.";
        append({ id: nextId(), role: "assistant", text: message, error: true });
      } finally {
        setAsking(false);
      }
    },
    [asking, append]
  );

  const clear = useCallback(() => setMessages([]), []);

  return { messages, asking, ask, clear };
}
