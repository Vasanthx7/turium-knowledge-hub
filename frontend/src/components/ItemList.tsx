// Read-only table of saved items.

import type { Item } from "../types";
import { LinkIcon, NoteIcon } from "./icons";
import { Badge, Spinner } from "./ui";

interface Props {
  items: Item[];
  loading: boolean;
  onSelect: (id: string) => void;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  const today = new Date();
  const sameDay = d.toDateString() === today.toDateString();
  if (sameDay) return `Today ${d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function ItemList({ items, loading, onSelect }: Props) {
  if (loading)
    return (
      <div className="p-6">
        <Spinner label="Loading items…" />
      </div>
    );

  if (items.length === 0)
    return (
      <div className="p-8 text-center text-sm text-slate-500">
        Nothing saved yet. Add a note or URL above to build your knowledge base.
      </div>
    );

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b border-white/10 text-xs uppercase tracking-wide text-slate-500">
            <th className="px-4 py-2 font-medium">Name</th>
            <th className="px-4 py-2 font-medium">Type</th>
            <th className="px-4 py-2 text-right font-medium">Added</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr
              key={item.id}
              onClick={() => onSelect(item.id)}
              className="cursor-pointer border-b border-white/5 align-top last:border-0 hover:bg-white/5"
            >
              <td className="px-4 py-3">
                <div className="flex items-start gap-2.5">
                  <span
                    className={`mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg ${
                      item.source_type === "url"
                        ? "bg-violet-500/15 text-violet-300"
                        : "bg-white/10 text-slate-400"
                    }`}
                  >
                    {item.source_type === "url" ? (
                      <LinkIcon className="h-4 w-4" />
                    ) : (
                      <NoteIcon className="h-4 w-4" />
                    )}
                  </span>
                  <div className="min-w-0">
                    <div className="truncate font-medium text-slate-200">
                      {item.title}
                    </div>
                    <div className="line-clamp-1 text-xs text-slate-500">
                      {item.source_url ?? item.preview}
                    </div>
                  </div>
                </div>
              </td>
              <td className="px-4 py-3">
                <Badge tone={item.source_type === "url" ? "violet" : "slate"}>
                  {item.source_type}
                </Badge>
              </td>
              <td className="whitespace-nowrap px-4 py-3 text-right text-xs text-slate-500">
                {formatDate(item.created_at)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
