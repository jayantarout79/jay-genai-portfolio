"use client";

import NextImage from "next/image";
import { useEffect, useMemo, useState } from "react";
import IOSCard from "@/components/ios-card";
import LargeTitleHeader from "@/components/large-title-header";
import LoadingSkeleton from "@/components/loading-skeleton";
import ModalSheet from "@/components/modal-sheet";
import PillBadge from "@/components/pill-badge";
import ScoreRing from "@/components/score-ring";
import Toast from "@/components/toast";
import { useDeviceId } from "@/hooks/use-device-id";
import type { AnalysisRecord, ScoreResult } from "@/lib/types";

const SLOT_LABELS = [
  "Front label (optional)",
  "Ingredients label (recommended)",
];

export default function AnalyzePage() {
  const { deviceId, ready } = useDeviceId();
  const [files, setFiles] = useState<(File | null)[]>([null, null]);
  const [previews, setPreviews] = useState<(string | null)[]>([null, null]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [toast, setToast] = useState<{ message: string | null; tone?: "neutral" | "success" | "error" }>({
    message: null,
  });
  const [analysis, setAnalysis] = useState<AnalysisRecord | null>(null);
  const [showLimitModal, setShowLimitModal] = useState(false);
  const [ocrStatus, setOcrStatus] = useState<string | null>(null);

  useEffect(() => {
    return () => {
      previews.forEach((url) => url && URL.revokeObjectURL(url));
    };
  }, [previews]);

  const handleFileChange = (index: number, file: File | null) => {
    setFiles((prev) => {
      const next = [...prev];
      next[index] = file;
      return next;
    });
    setPreviews((prev) => {
      const next = [...prev];
      if (prev[index]) URL.revokeObjectURL(prev[index]!);
      next[index] = file ? URL.createObjectURL(file) : null;
      return next;
    });
  };

  const handleSubmit = async () => {
    if (!ready || !deviceId) {
      setToast({ message: "Device ID not ready", tone: "error" });
      return;
    }
    if (!files.filter(Boolean).length) {
      setToast({ message: "Add at least one image", tone: "error" });
      return;
    }

    setIsSubmitting(true);
    setToast({ message: null });

    try {
      const optimizedFiles = await Promise.all(
        files.map(async (file) => (file ? await downscaleImage(file) : null)),
      );
      const presentFiles = optimizedFiles.filter(Boolean) as File[];

      if (!presentFiles.length) {
        throw new Error("No usable images found");
      }

      setOcrStatus("Reading label locally…");
      const extractedText = await runLocalOcr(presentFiles);
      setOcrStatus(null);

      const form = new FormData();
      form.append("device_id", deviceId);
      if (extractedText.trim()) form.append("extracted_text", extractedText.trim());
      presentFiles.forEach((file) => form.append("images", file));

      const res = await fetch("/api/analyze", {
        method: "POST",
        body: form,
      });

      if (res.status === 429) {
        setShowLimitModal(true);
        return;
      }

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err?.error || "Analyze failed");
      }

      const payload = (await res.json()) as AnalysisRecord;
      setAnalysis(payload);
      setToast({ message: "Analysis ready", tone: "success" });
    } catch (error) {
      console.error(error);
      setToast({ message: error instanceof Error ? error.message : "Analyze failed", tone: "error" });
    } finally {
      setIsSubmitting(false);
      setOcrStatus(null);
    }
  };

  const handleReset = () => {
    setFiles([null, null]);
    previews.forEach((url) => url && URL.revokeObjectURL(url));
    setPreviews([null, null]);
    setAnalysis(null);
    setToast({ message: null });
  };

  const score: ScoreResult | null = useMemo(() => {
    if (!analysis) return null;
    return analysis.analysis_result;
  }, [analysis]);

  return (
    <div className="pb-24">
      <LargeTitleHeader
        title="Analyze"
        subtitle="Upload up to 2 photos for an instant verdict. Images delete right after processing."
      />

      <div className="space-y-4">
        <IOSCard>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">Upload</h3>
              <PillBadge tone="warn">Max 8MB · 2 photos</PillBadge>
            </div>

            <div className="grid grid-cols-1 gap-3">
              {SLOT_LABELS.map((label, idx) => (
                <label
                  key={label}
                  className="flex cursor-pointer items-center justify-between rounded-2xl border border-dashed border-white/20 bg-white/5 px-4 py-3 transition hover:border-white/40"
                >
                  <div>
                    <p className="text-sm font-semibold text-white">{label}</p>
                    <p className="text-xs text-slate-400">JPEG/PNG/HEIC · camera capture OK</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {previews[idx] ? (
                      <NextImage
                        src={previews[idx] as string}
                        alt={label}
                        width={56}
                        height={56}
                        className="h-14 w-14 rounded-xl object-cover shadow-lg"
                        unoptimized
                      />
                    ) : (
                      <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-white/10 text-xl text-white/70">
                        +
                      </div>
                    )}
                    <input
                      type="file"
                      accept="image/jpeg,image/png,image/heic,image/heif"
                      capture="environment"
                      className="hidden"
                      onChange={(e) => handleFileChange(idx, e.target.files?.[0] ?? null)}
                    />
                  </div>
                </label>
              ))}
            </div>

            <button
              onClick={handleSubmit}
              disabled={isSubmitting}
              className="w-full rounded-2xl bg-sky-500 px-4 py-3 text-center text-white font-semibold shadow-lg transition hover:bg-sky-600 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {isSubmitting ? "Analyzing..." : "Analyze"}
            </button>
            <button
              onClick={handleReset}
              className="w-full rounded-2xl border border-white/15 px-4 py-3 text-center text-white font-semibold transition hover:border-white/40"
            >
              Reset
            </button>
            <p className="text-xs text-slate-400">
              No accounts. Limit 5 analyses per day. Images are deleted immediately after processing.
            </p>
          </div>
        </IOSCard>

        {isSubmitting && (
          <IOSCard>
            <p className="text-sm text-slate-300">{ocrStatus || "Processing…"}</p>
            <div className="mt-4 flex items-center justify-center">
              <div className="h-12 w-12 rounded-full border-2 border-white/30 border-t-white animate-spin" />
            </div>
            <div className="mt-4">
              <LoadingSkeleton lines={3} />
            </div>
          </IOSCard>
        )}

        {analysis && score && (
          <IOSCard className="space-y-4">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm text-slate-400">Result</p>
                <h3 className="text-xl font-semibold text-white">
                  {analysis.product_name?.trim() || "Unknown product"}
                </h3>
                <p className="text-sm text-slate-300">{analysis.brand?.trim() || "Unknown brand"}</p>
                <div className="mt-2 flex flex-wrap gap-2">
                  <PillBadge tone={score.buy_score_percent >= 70 ? "success" : score.buy_score_percent >= 45 ? "warn" : "danger"}>
                    {score.verdict}
                  </PillBadge>
                  <PillBadge tone="neutral">Confidence: {(analysis.confidence ?? 0).toFixed(2)}</PillBadge>
                </div>
              </div>
              <ScoreRing value={score.buy_score_percent} label="Buy score" />
            </div>

            <div className="grid grid-cols-1 gap-3">
              <DetailBlock title="Good ingredients" items={score.good_ingredients.map((g) => `${g.name} — ${g.why}`)} tone="success" />
              <DetailBlock title="Concerning" items={score.concerning_ingredients.map((g) => `${g.name} — ${g.why}`)} tone="danger" />
              <DetailBlock title="Allergens/irritants" items={score.allergens_or_irritants.map((a) => `${a.name} — ${a.who_should_avoid}`)} tone="warn" />
              <DetailBlock title="Warnings" items={score.warnings} tone="warn" />
              <DetailBlock title="Data quality" items={[score.data_quality_notes || ""]} tone="neutral" />
            </div>

            <p className="text-xs text-slate-400">
              Summary: {score.summary || "No summary returned."} — Informational only. Not medical advice.
            </p>
          </IOSCard>
        )}
      </div>

      <Toast message={toast.message} tone={toast.tone} />

      <ModalSheet
        open={showLimitModal}
        title="Daily limit reached"
        description="You can run up to 5 analyses per day per device ID."
        confirmLabel="Got it"
        onCancel={() => setShowLimitModal(false)}
        onConfirm={() => setShowLimitModal(false)}
      />
    </div>
  );
}

async function downscaleImage(file: File, maxSize = 1200): Promise<File> {
  if (typeof window === "undefined" || !file.type.startsWith("image/")) return file;
  const blobUrl = URL.createObjectURL(file);
  const img = new window.Image();
  const loaded = new Promise<HTMLImageElement>((resolve, reject) => {
    img.onload = () => resolve(img);
    img.onerror = reject;
  });
  img.src = blobUrl;

  try {
    const image = await loaded;
    const maxDimension = Math.max(image.width, image.height);
    if (!maxDimension || maxDimension <= maxSize) {
      return file;
    }

    const scale = maxSize / maxDimension;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(image.width * scale);
    canvas.height = Math.round(image.height * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return file;
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    const blob: Blob | null = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.82),
    );
    if (!blob) return file;
    return new File([blob], file.name.replace(/\.(heic|heif)$/i, ".jpg"), { type: "image/jpeg" });
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

async function runLocalOcr(files: File[]): Promise<string> {
  if (!files.length) return "";
  try {
    const Tesseract = await import("tesseract.js");
    const texts: string[] = [];
    for (const file of files) {
      const result = await Tesseract.recognize(file, "eng");
      const text = result?.data?.text ?? "";
      if (text.trim()) texts.push(text.trim());
    }
    return texts.join("\n").trim();
  } catch (error) {
    console.error("Local OCR failed", error);
    return "";
  }
}

function DetailBlock({
  title,
  items,
  tone,
}: {
  title: string;
  items: string[];
  tone: "success" | "danger" | "warn" | "neutral";
}) {
  const filtered = items.filter(Boolean);
  if (!filtered.length) return null;

  return (
    <div className="rounded-2xl bg-white/5 p-3">
      <div className="mb-1 flex items-center gap-2">
        <PillBadge tone={tone}>{title}</PillBadge>
      </div>
      <ul className="list-disc space-y-1 pl-4 text-sm text-slate-200">
        {filtered.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}
