import type { PropsWithChildren } from "react";

export default function IOSScreenFrame({ children }: PropsWithChildren) {
  return (
    <div className="w-full">
      <div className="mx-auto flex max-w-[430px] flex-col rounded-[32px] border border-white/10 bg-gradient-to-b from-white/5 to-white/0 p-0 shadow-[0_20px_80px_rgba(0,0,0,0.55)] md:rounded-[40px]">
        <div className="relative overflow-hidden rounded-[32px] md:rounded-[40px]">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(34,211,238,0.15),transparent_30%),radial-gradient(circle_at_80%_0%,rgba(168,85,247,0.12),transparent_28%)]" />
          <div className="relative min-h-screen md:min-h-[760px] bg-[rgba(9,12,26,0.88)] px-4 pb-24 pt-4 md:px-5 md:pt-6">
            {children}
          </div>
        </div>
      </div>
    </div>
  );
}
