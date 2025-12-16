"use client";

import { AnimatePresence, motion } from "framer-motion";
import { usePathname } from "next/navigation";
import type { PropsWithChildren } from "react";
import IOSScreenFrame from "./ios-screen-frame";
import IOSTabBar from "./ios-tab-bar";

const tabs = [
  { href: "/analyze", label: "Analyze", icon: "🔍" },
];

export default function AppShell({ children }: PropsWithChildren) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen w-full">
      <div className="flex min-h-screen items-center justify-center px-3 py-4 md:px-6">
        <IOSScreenFrame>
          <div className="flex min-h-screen md:min-h-[760px] flex-col">
            <AnimatePresence mode="wait">
              <motion.main
                key={pathname}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.18 }}
                className="flex-1"
              >
                {children}
              </motion.main>
            </AnimatePresence>
            <IOSTabBar currentPath={pathname} tabs={tabs} />
          </div>
        </IOSScreenFrame>
      </div>
    </div>
  );
}
