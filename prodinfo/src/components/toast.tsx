import { AnimatePresence, motion } from "framer-motion";

type Props = {
  message: string | null;
  tone?: "neutral" | "success" | "error";
};

export default function Toast({ message, tone = "neutral" }: Props) {
  const bg =
    tone === "success"
      ? "bg-emerald-500/90 text-white"
      : tone === "error"
        ? "bg-rose-500/90 text-white"
        : "bg-slate-800/90 text-white";

  return (
    <AnimatePresence>
      {message ? (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 10 }}
          className="fixed bottom-24 left-1/2 z-50 -translate-x-1/2"
        >
          <div className={`rounded-full px-4 py-2 text-sm shadow-lg ${bg}`}>{message}</div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
