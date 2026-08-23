// Add-knowledge card with a note/URL toggle; saving is delegated via props.

import { useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { LinkIcon, NoteIcon } from "./icons";
import { ErrorBanner, Spinner } from "./ui";

type Mode = "note" | "url";

interface Props {
  submitting: boolean;
  error: string | null;
  onAddNote: (text: string, title?: string) => Promise<boolean>;
  onAddUrl: (url: string) => Promise<boolean>;
}

const INPUT =
  "w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 outline-none transition focus:border-violet-500/70 focus:bg-white/[0.07]";

export function AddItemForm({ submitting, error, onAddNote, onAddUrl }: Props) {
  const [mode, setMode] = useState<Mode>("note");
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");

  const canSubmit =
    !submitting &&
    (mode === "note" ? text.trim().length > 0 : url.trim().length > 0);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    const ok =
      mode === "note"
        ? await onAddNote(text.trim(), title.trim() || undefined)
        : await onAddUrl(url.trim());
    if (ok) {
      setText("");
      setTitle("");
      setUrl("");
    }
  }

  const tabs: { mode: Mode; label: string; icon: ReactNode }[] = [
    { mode: "note", label: "Note", icon: <NoteIcon className="h-4 w-4" /> },
    { mode: "url", label: "URL", icon: <LinkIcon className="h-4 w-4" /> },
  ];

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-xl border border-white/10 bg-white/[0.03] p-4"
    >
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-200">Add Knowledge</h2>
        <div className="inline-flex rounded-lg border border-white/10 bg-white/5 p-0.5">
          {tabs.map((t) => (
            <button
              key={t.mode}
              type="button"
              onClick={() => setMode(t.mode)}
              className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition ${
                mode === t.mode
                  ? "bg-violet-600 text-white shadow-sm shadow-violet-900/40"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {t.icon}
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {mode === "note" ? (
        <div className="space-y-2">
          <input
            className={INPUT}
            placeholder="Title (optional)"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
          <textarea
            className={`${INPUT} h-28 resize-y`}
            placeholder="Paste or type a note to remember…"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </div>
      ) : (
        <input
          className={INPUT}
          placeholder="https://example.com/article"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
      )}

      {error && (
        <div className="mt-2">
          <ErrorBanner message={error} />
        </div>
      )}

      <button
        type="submit"
        disabled={!canSubmit}
        className="mt-3 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-violet-600 px-4 py-2 text-sm font-medium text-white shadow-lg shadow-violet-900/30 transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {submitting ? (
          <Spinner label="Saving…" />
        ) : mode === "note" ? (
          "Save note"
        ) : (
          "Fetch & save URL"
        )}
      </button>
    </form>
  );
}
