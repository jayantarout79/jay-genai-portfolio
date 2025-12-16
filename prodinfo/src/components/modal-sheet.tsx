import { AnimatePresence, motion } from "framer-motion";
import type { PropsWithChildren } from "react";

type Props = PropsWithChildren & {
  open: boolean;
  title: string;
  description?: string;
  confirmLabel?: string;
  destructive?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
};

export default function ModalSheet({
  open,
  title,
  description,
  confirmLabel = "Confirm",
  destructive,
  onConfirm,
  onCancel,
  children,
}: Props) {
  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 px-4 pb-10"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onCancel}
        >
          <motion.div
            initial={{ y: 40 }}
            animate={{ y: 0 }}
            exit={{ y: 60 }}
            transition={{ type: "spring", stiffness: 260, damping: 22 }}
            className="glass-surface w-full max-w-[430px] rounded-3xl bg-slate-900/90 p-5 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-3 h-1 w-16 self-center rounded-full bg-white/15" />
            <h3 className="text-xl font-semibold text-white">{title}</h3>
            {description && <p className="mt-1 text-slate-300">{description}</p>}
            {children}
            <div className="mt-4 flex gap-3">
              <button
                className="flex-1 rounded-full border border-white/20 px-4 py-3 text-white transition hover:border-white/40"
                onClick={onCancel}
              >
                Cancel
              </button>
              <button
                className={`flex-1 rounded-full px-4 py-3 font-semibold text-white transition ${
                  destructive
                    ? "bg-rose-500 hover:bg-rose-600"
                    : "bg-sky-500 hover:bg-sky-600"
                }`}
                onClick={onConfirm}
              >
                {confirmLabel}
              </button>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
