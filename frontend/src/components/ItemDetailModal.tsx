// Modal to view, edit, and delete a saved item; calls onChanged after a mutation so the parent refreshes.

import { useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { ItemDetail } from "../types";
import { LinkIcon, NoteIcon } from "./icons";
import { Badge, ErrorBanner, Spinner } from "./ui";

interface Props {
  itemId: string;
  onClose: () => void;
  onChanged: () => void;
}

export function ItemDetailModal({ itemId, onClose, onChanged }: Props) {
  const [item, setItem] = useState<ItemDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<"view" | "edit">("view");
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [busy, setBusy] = useState(false);

  // Load the item's full content when the modal opens.
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    api
      .getItem(itemId)
      .then((detail) => {
        if (!active) return;
        setItem(detail);
        setTitle(detail.title);
        setContent(detail.content);
      })
      .catch((e) => {
        if (active) setError(e instanceof ApiError ? e.message : "Failed to load.");
      })
      .finally(() => active && setLoading(false));
    return () => {
      active = false;
    };
  }, [itemId]);

  // Close on Escape.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const handleSave = useCallback(async () => {
    if (!item) return;
    setBusy(true);
    setError(null);
    try {
      const updated = await api.updateItem(item.id, {
        title,
        content: content !== item.content ? content : undefined,
      });
      setItem(updated);
      setTitle(updated.title);
      setContent(updated.content);
      setMode("view");
      onChanged();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to save changes.");
    } finally {
      setBusy(false);
    }
  }, [item, title, content, onChanged]);

  const handleDelete = useCallback(async () => {
    if (!item) return;
    if (!window.confirm(`Delete "${item.title}"? This cannot be undone.`)) return;
    setBusy(true);
    setError(null);
    try {
      await api.deleteItem(item.id);
      onChanged();
      onClose();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to delete.");
      setBusy(false);
    }
  }, [item, onChanged, onClose]);

  const canSave = mode === "edit" && !busy && content.trim().length > 0;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#12121b] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-3 border-b border-white/10 px-5 py-4">
          <div className="flex min-w-0 items-center gap-2.5">
            <span
              className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${
                item?.source_type === "url"
                  ? "bg-violet-500/15 text-violet-300"
                  : "bg-white/10 text-slate-400"
              }`}
            >
              {item?.source_type === "url" ? (
                <LinkIcon className="h-4 w-4" />
              ) : (
                <NoteIcon className="h-4 w-4" />
              )}
            </span>
            <div className="min-w-0">
              {mode === "view" ? (
                <h2 className="truncate text-base font-semibold text-slate-100">
                  {item?.title ?? "Loading…"}
                </h2>
              ) : (
                <span className="text-sm font-medium text-slate-400">
                  Editing
                </span>
              )}
              {item && (
                <div className="mt-0.5 flex items-center gap-2 text-xs text-slate-500">
                  <Badge tone={item.source_type === "url" ? "violet" : "slate"}>
                    {item.source_type}
                  </Badge>
                  <span>{new Date(item.created_at).toLocaleString()}</span>
                </div>
              )}
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="rounded-md p-1 text-slate-500 transition hover:bg-white/10 hover:text-slate-200"
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4">
          {loading ? (
            <Spinner label="Loading…" />
          ) : error && !item ? (
            <ErrorBanner message={error} />
          ) : item ? (
            <div className="space-y-3">
              {item.source_url && (
                <a
                  href={item.source_url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-block truncate text-sm text-violet-400 hover:underline"
                >
                  {item.source_url}
                </a>
              )}

              {mode === "view" ? (
                <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-300">
                  {item.content}
                </p>
              ) : (
                <div className="space-y-2">
                  <input
                    className="w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none transition focus:border-violet-500/70"
                    placeholder="Title"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                  />
                  <textarea
                    className="h-64 w-full resize-y rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm leading-relaxed text-slate-100 outline-none transition focus:border-violet-500/70"
                    value={content}
                    onChange={(e) => setContent(e.target.value)}
                  />
                  <p className="text-xs text-slate-500">
                    Editing the content re-chunks and re-embeds this item.
                  </p>
                </div>
              )}

              {error && item && <ErrorBanner message={error} />}
            </div>
          ) : null}
        </div>

        {/* Footer actions */}
        {item && (
          <div className="flex items-center justify-between gap-2 border-t border-white/10 px-5 py-3">
            <button
              onClick={handleDelete}
              disabled={busy}
              className="rounded-lg px-3 py-2 text-sm font-medium text-red-400 transition hover:bg-red-500/10 disabled:opacity-50"
            >
              Delete
            </button>

            {mode === "view" ? (
              <button
                onClick={() => setMode("edit")}
                className="rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-violet-500"
              >
                Edit
              </button>
            ) : (
              <div className="flex gap-2">
                <button
                  onClick={() => {
                    setMode("view");
                    setTitle(item.title);
                    setContent(item.content);
                    setError(null);
                  }}
                  disabled={busy}
                  className="rounded-lg px-4 py-2 text-sm font-medium text-slate-400 transition hover:bg-white/10 disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSave}
                  disabled={!canSave}
                  className="inline-flex items-center gap-2 rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {busy ? <Spinner label="Saving…" /> : "Save changes"}
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
