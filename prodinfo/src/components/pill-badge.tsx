import clsx from "clsx";
import type { PropsWithChildren } from "react";

type Props = PropsWithChildren & {
  tone?: "neutral" | "success" | "warn" | "danger";
};

export default function PillBadge({ tone = "neutral", children }: Props) {
  const toneClass =
    tone === "success"
      ? "bg-emerald-400/15 text-emerald-300 border-emerald-200/30"
      : tone === "warn"
        ? "bg-amber-400/15 text-amber-200 border-amber-200/30"
        : tone === "danger"
          ? "bg-rose-400/15 text-rose-200 border-rose-200/30"
          : "bg-white/10 text-slate-100 border-white/20";

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full border px-3 py-1 text-xs font-semibold",
        toneClass,
      )}
    >
      {children}
    </span>
  );
}
