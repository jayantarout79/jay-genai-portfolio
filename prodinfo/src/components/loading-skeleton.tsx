export default function LoadingSkeleton({ lines = 3 }: { lines?: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: lines }).map((_, idx) => (
        <div
          // biome-ignore lint/suspicious/noArrayIndexKey: static skeleton
          key={idx}
          className="h-3 animate-pulse rounded-full bg-white/10"
          style={{ width: `${80 + (idx % 2 === 0 ? 10 : 0)}%` }}
        />
      ))}
    </div>
  );
}
