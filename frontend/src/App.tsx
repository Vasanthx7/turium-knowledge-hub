// Two-pane layout: Knowledge column (left) and RAG chat panel (right).
// Sharing `items` keeps the list and the chat's indexed count in sync.

import { useState } from "react";
import { AddItemForm } from "./components/AddItemForm";
import { ChatPanel } from "./components/ChatPanel";
import { ItemDetailModal } from "./components/ItemDetailModal";
import { ItemList } from "./components/ItemList";
import { useChat } from "./hooks/useChat";
import { useItems } from "./hooks/useItems";

export default function App() {
  const { items, loading, error, submitting, addNote, addUrl, reload } =
    useItems();
  const { messages, asking, ask } = useChat();
  const [selectedId, setSelectedId] = useState<string | null>(null);

  return (
    <div className="relative z-10 flex h-screen flex-col">
      <header className="flex items-center gap-3 border-b border-white/10 bg-[#0d0d17]/70 px-5 py-3 backdrop-blur">
        <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 text-sm font-bold text-white shadow-lg shadow-violet-900/40">
          T
        </span>
        <div className="leading-tight">
          <h1 className="text-sm font-semibold tracking-tight text-white">
            Turium <span className="text-slate-400">Knowledge Hub</span>
          </h1>
        </div>
      </header>

      <main className="grid flex-1 grid-cols-1 overflow-hidden md:grid-cols-[1.05fr_1fr]">
        <section className="flex flex-col gap-4 overflow-y-auto border-white/10 p-5 md:border-r">
          <AddItemForm
            submitting={submitting}
            error={error}
            onAddNote={addNote}
            onAddUrl={addUrl}
          />

          <div className="overflow-hidden rounded-xl border border-white/10 bg-white/[0.03]">
            <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
              <h2 className="text-sm font-semibold text-slate-200">
                Knowledge Base
              </h2>
              <span className="rounded-full bg-white/10 px-2 py-0.5 text-xs font-medium text-slate-400">
                {items.length} item{items.length === 1 ? "" : "s"}
              </span>
            </div>
            <ItemList
              items={items}
              loading={loading}
              onSelect={setSelectedId}
            />
          </div>
        </section>

        <section className="min-h-0 border-white/10">
          <ChatPanel
            messages={messages}
            asking={asking}
            itemCount={items.length}
            onAsk={ask}
          />
        </section>
      </main>

      {selectedId && (
        <ItemDetailModal
          itemId={selectedId}
          onClose={() => setSelectedId(null)}
          onChanged={reload}
        />
      )}
    </div>
  );
}
