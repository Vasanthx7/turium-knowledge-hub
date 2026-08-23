// Saved-items state: list, loading/error flags, and the ingest actions.

import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { Item } from "../types";

export interface UseItems {
  items: Item[];
  loading: boolean;
  error: string | null;
  submitting: boolean;
  addNote: (text: string, title?: string) => Promise<boolean>;
  addUrl: (url: string) => Promise<boolean>;
  reload: () => Promise<void>;
}

export function useItems(): UseItems {
  const [items, setItems] = useState<Item[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const { items } = await api.listItems();
      setItems(items);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load items.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  // Shared ingest wrapper; reloads on success. Returns success so the form
  // can decide whether to clear its inputs.
  const ingest = useCallback(
    async (action: () => Promise<unknown>): Promise<boolean> => {
      setSubmitting(true);
      setError(null);
      try {
        await action();
        await reload();
        return true;
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Failed to save.");
        return false;
      } finally {
        setSubmitting(false);
      }
    },
    [reload]
  );

  const addNote = useCallback(
    (text: string, title?: string) => ingest(() => api.ingestNote(text, title)),
    [ingest]
  );

  const addUrl = useCallback(
    (url: string) => ingest(() => api.ingestUrl(url)),
    [ingest]
  );

  return { items, loading, error, submitting, addNote, addUrl, reload };
}
