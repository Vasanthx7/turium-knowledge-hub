// Shared presentational primitives for the dark theme.

import type { ReactNode } from "react";

export function Card({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="rounded-xl border border-white/10 bg-white/[0.03]">
      <h2 className="border-b border-white/10 px-5 py-3 text-sm font-semibold uppercase tracking-wide text-slate-400">
        {title}
      </h2>
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-slate-400">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-violet-400" />
      {label}
    </span>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
      {message}
    </div>
  );
}

export function Badge({
  children,
  tone = "slate",
}: {
  children: ReactNode;
  tone?: "slate" | "violet" | "green";
}) {
  const tones = {
    slate: "bg-white/10 text-slate-300",
    violet: "bg-violet-500/15 text-violet-300",
    green: "bg-emerald-500/15 text-emerald-300",
  };
  return (
    <span
      className={`rounded-full px-2 py-0.5 text-xs font-medium ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function Dot({ className = "bg-emerald-400" }: { className?: string }) {
  return (
    <span className="relative flex h-2 w-2">
      <span
        className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-70 ${className}`}
      />
      <span className={`relative inline-flex h-2 w-2 rounded-full ${className}`} />
    </span>
  );
}
