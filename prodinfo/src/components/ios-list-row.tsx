import clsx from "clsx";
import type { PropsWithChildren, ReactNode } from "react";

type Props = PropsWithChildren & {
  leading?: ReactNode;
  trailing?: ReactNode;
  onClick?: () => void;
  href?: string;
};

export default function IOSListRow({ leading, trailing, children }: Props) {
  return (
    <div className="flex items-center justify-between gap-4 py-3">
      <div className="flex items-center gap-3">
        {leading && <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white/5 text-lg">{leading}</div>}
        <div className="text-sm font-medium text-white">{children}</div>
      </div>
      {trailing && <div className={clsx("text-right text-sm text-slate-300")}>{trailing}</div>}
    </div>
  );
}
