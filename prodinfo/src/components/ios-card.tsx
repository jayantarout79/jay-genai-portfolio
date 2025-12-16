import clsx from "clsx";
import type { PropsWithChildren } from "react";

type Props = PropsWithChildren & {
  className?: string;
  onClick?: () => void;
};

export default function IOSCard({ children, className, onClick }: Props) {
  return (
    <div
      onClick={onClick}
      className={clsx(
        "card relative overflow-hidden rounded-2xl border border-white/10 bg-white/5 p-4 transition hover:border-white/20",
        onClick && "cursor-pointer active:scale-[0.99]",
        className,
      )}
    >
      <div className="pointer-events-none absolute inset-0 rounded-2xl border border-white/5" />
      {children}
    </div>
  );
}
