import clsx from "clsx";
import Link from "next/link";

type Tab = {
  href: string;
  label: string;
  icon: string;
};

export default function IOSTabBar({
  currentPath,
  tabs,
}: {
  currentPath: string;
  tabs: Tab[];
}) {
  if (!tabs.length) return null;
  return (
    <div className="fixed inset-x-0 bottom-0 z-40 mx-auto w-full max-w-[430px] px-4 pb-[calc(env(safe-area-inset-bottom)+12px)]">
      <div className="glass-surface flex items-center justify-between rounded-full px-3 py-2">
        {tabs.map((tab) => {
          const isActive =
            tab.href === "/"
              ? currentPath === "/"
              : currentPath.startsWith(tab.href);
          return (
            <Link
              key={tab.href}
              href={tab.href}
              className={clsx(
                "flex flex-1 flex-col items-center gap-1 rounded-full px-2 py-1 text-xs font-semibold transition-all duration-200",
                isActive
                  ? "text-white"
                  : "text-slate-300 hover:text-white hover:translate-y-[-1px]",
              )}
            >
              <span
                aria-hidden
                className={clsx(
                  "text-lg transition-transform duration-200",
                  isActive ? "scale-110" : "scale-100 opacity-80",
                )}
              >
                {tab.icon}
              </span>
              <span className="leading-none">{tab.label}</span>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
