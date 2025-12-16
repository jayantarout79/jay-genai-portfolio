type Props = {
  title: string;
  subtitle?: string;
  trailing?: React.ReactNode;
};

export default function LargeTitleHeader({ title, subtitle, trailing }: Props) {
  return (
    <div className="mb-6 flex items-start justify-between gap-3">
      <div>
        <p className="text-sm uppercase tracking-[0.12em] text-slate-400">
          IngrediScore Web
        </p>
        <h1 className="text-4xl font-semibold text-white md:text-5xl">{title}</h1>
        {subtitle && <p className="mt-1 text-slate-300">{subtitle}</p>}
      </div>
      {trailing}
    </div>
  );
}
