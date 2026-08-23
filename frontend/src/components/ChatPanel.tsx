// RAG chat panel: status header, scrolling message thread, and input bar.

import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import type { ChatMessage, Citation } from "../types";
import { BotIcon, InfoIcon, LinkIcon, NoteIcon, SendIcon } from "./icons";
import { Badge, Dot } from "./ui";

interface Props {
  messages: ChatMessage[];
  asking: boolean;
  itemCount: number;
  onAsk: (question: string) => void;
}

export function ChatPanel({ messages, asking, itemCount, onAsk }: Props) {
  const [question, setQuestion] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  // Keep the latest turn in view as the conversation grows.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, asking]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (q.length < 3 || asking) return;
    onAsk(q);
    setQuestion("");
  }

  return (
    <div className="flex h-full flex-col bg-white/[0.02]">
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-white/10 px-5 py-3">
        <Avatar size="lg" />
        <div>
          <div className="text-sm font-semibold text-slate-100">
            Knowledge Assistant
          </div>
          <div className="flex items-center gap-1.5 text-xs text-slate-500">
            <Dot />
            RAG Active · {itemCount} item{itemCount === 1 ? "" : "s"} indexed
          </div>
        </div>
      </div>

      {/* Thread */}
      <div className="flex-1 space-y-4 overflow-y-auto px-5 py-5">
        <GreetingBubble />
        {messages.map((m) =>
          m.role === "user" ? (
            <UserBubble key={m.id} text={m.text} />
          ) : (
            <AssistantBubble key={m.id} message={m} />
          )
        )}
        {asking && <TypingBubble />}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-white/10 px-5 py-3">
        <form onSubmit={handleSubmit} className="relative">
          <input
            className="w-full rounded-xl border border-white/10 bg-white/5 py-3 pl-4 pr-12 text-sm text-slate-100 placeholder-slate-500 outline-none transition focus:border-violet-500/70 focus:bg-white/[0.07]"
            placeholder="Ask a question about your saved content…"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button
            type="submit"
            disabled={asking || question.trim().length < 3}
            aria-label="Send"
            className="absolute right-2 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-lg bg-violet-600 text-white transition hover:bg-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            <SendIcon className="h-4 w-4" />
          </button>
        </form>
        <p className="mt-2 text-center text-[11px] text-slate-600">
          Answers are grounded in your saved content. Verify important details.
        </p>
      </div>
    </div>
  );
}

// Message pieces
function GreetingBubble() {
  return (
    <div className="flex gap-3">
      <Avatar />
      <div className="max-w-[85%] rounded-2xl rounded-tl-sm border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-300">
        Hi! Ask me anything about the notes and pages you've saved. I'll answer
        using only your content and cite the sources.
      </div>
    </div>
  );
}

function UserBubble({ text }: { text: string }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-tr-sm bg-violet-600 px-4 py-3 text-sm text-white shadow-lg shadow-violet-900/30">
        {text}
      </div>
    </div>
  );
}

function AssistantBubble({ message }: { message: ChatMessage }) {
  const hasCitations = !!message.citations && message.citations.length > 0;
  // No error but no citations means the relevance gate declined: info note, not an error.
  const isNoAnswer = !message.error && !hasCitations;

  if (isNoAnswer) {
    return (
      <div className="flex gap-3">
        <Avatar />
        <div className="max-w-[85%] rounded-2xl rounded-tl-sm border border-amber-400/25 bg-amber-400/[0.07] px-4 py-3 text-sm text-amber-100/90">
          <div className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-amber-300/80">
            <InfoIcon className="h-3.5 w-3.5" />
            No grounded answer
          </div>
          <p className="whitespace-pre-wrap leading-relaxed">{message.text}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      <Avatar />
      <div className="max-w-[85%] space-y-3">
        <div
          className={`rounded-2xl rounded-tl-sm border px-4 py-3 text-sm ${
            message.error
              ? "border-red-500/30 bg-red-500/10 text-red-300"
              : "border-white/10 bg-white/5 text-slate-200"
          }`}
        >
          <p className="whitespace-pre-wrap leading-relaxed">{message.text}</p>
        </div>
        {hasCitations && <CitationList citations={message.citations!} />}
      </div>
    </div>
  );
}

const SOURCE_LABEL: Record<Citation["source_type"], string> = {
  note: "Note",
  url: "Web page",
};

function CitationList({ citations }: { citations: Citation[] }) {
  return (
    <div className="space-y-1.5">
      <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
        {citations.length} Source{citations.length === 1 ? "" : "s"}
      </div>
      {citations.map((c, i) => (
        <div
          key={`${c.item_id}-${i}`}
          className="rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 text-xs"
        >
          <div className="flex items-center gap-2">
            <Badge tone="violet">{i + 1}</Badge>
            <span className="flex min-w-0 flex-1 items-center gap-1.5 font-medium text-slate-300">
              {c.source_type === "url" ? (
                <LinkIcon className="h-3.5 w-3.5 shrink-0 text-violet-400" />
              ) : (
                <NoteIcon className="h-3.5 w-3.5 shrink-0 text-slate-500" />
              )}
              <span className="truncate">{c.title}</span>
            </span>
            <span className="shrink-0 text-[10px] uppercase tracking-wide text-slate-600">
              {SOURCE_LABEL[c.source_type]}
            </span>
          </div>
          <p className="mt-1.5 line-clamp-2 text-slate-500">{c.snippet}</p>
          {c.source_url && (
            <a
              href={c.source_url}
              target="_blank"
              rel="noreferrer"
              className="mt-1 inline-block truncate text-violet-400 hover:underline"
            >
              {c.source_url}
            </a>
          )}
        </div>
      ))}
    </div>
  );
}

function TypingBubble() {
  return (
    <div className="flex gap-3">
      <Avatar />
      <div className="rounded-2xl rounded-tl-sm border border-white/10 bg-white/5 px-4 py-3">
        <span className="flex gap-1">
          {[0, 150, 300].map((delay) => (
            <span
              key={delay}
              className="h-2 w-2 animate-bounce rounded-full bg-violet-400"
              style={{ animationDelay: `${delay}ms` }}
            />
          ))}
        </span>
      </div>
    </div>
  );
}

function Avatar({ size = "sm" }: { size?: "sm" | "lg" }) {
  const dim = size === "lg" ? "h-9 w-9" : "h-8 w-8";
  return (
    <span
      className={`flex ${dim} shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 text-white shadow-lg shadow-violet-900/40`}
    >
      <BotIcon className={size === "lg" ? "h-5 w-5" : "h-4 w-4"} />
    </span>
  );
}
