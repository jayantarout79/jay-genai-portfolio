import { z } from "zod";
import { type ExtractionResult, type ScoreResult } from "./types";

function toStringArray(value: unknown) {
  if (Array.isArray(value)) return value;
  if (typeof value === "string") {
    return value
      .split(/[,;\n•]+/)
      .map((s) => s.trim())
      .filter(Boolean);
  }
  return [];
}

const nullableStringToString = z
  .union([z.string(), z.null(), z.undefined()])
  .transform((v) => (typeof v === "string" ? v : ""));

const categorySchema = z
  .union([z.literal("food"), z.literal("cosmetics"), z.literal("other"), z.string(), z.null(), z.undefined()])
  .transform((v) => (v === "food" || v === "cosmetics" ? v : "other" as const));

const numberish = z
  .union([z.number(), z.string(), z.null(), z.undefined()])
  .transform((v) => {
    if (typeof v === "number") return v;
    const parsed = typeof v === "string" ? Number.parseFloat(v) : NaN;
    return Number.isFinite(parsed) ? parsed : 0;
  });

export const extractionSchema = z.object({
  product_name: nullableStringToString,
  brand: nullableStringToString,
  category: categorySchema,
  ingredients: z.preprocess(toStringArray, z.array(z.string())).default([]),
  allergens_found: z.preprocess(toStringArray, z.array(z.string())).default([]),
  claims_on_label: z.preprocess(toStringArray, z.array(z.string())).default([]),
  confidence: numberish.pipe(z.number().min(0).max(1)).catch(0),
  unreadable_parts: z.preprocess(toStringArray, z.array(z.string())).default([]),
}).strict();

export const scoreSchema = z.object({
  buy_score_percent: numberish.pipe(z.number().min(0).max(100)).catch(0),
  verdict: nullableStringToString,
  good_ingredients: z
    .array(
      z.object({
        name: nullableStringToString,
        why: nullableStringToString,
        impact: z.enum(["low", "med", "high"]).default("low"),
      }),
    )
    .default([]),
  concerning_ingredients: z
    .array(
      z.object({
        name: nullableStringToString,
        why: nullableStringToString,
        risk: z.enum(["low", "med", "high"]).default("low"),
      }),
    )
    .default([]),
  allergens_or_irritants: z
    .array(
      z.object({
        name: nullableStringToString,
        who_should_avoid: nullableStringToString,
      }),
    )
    .default([]),
  additives_or_preservatives: z
    .array(
      z.object({
        name: nullableStringToString,
        note: nullableStringToString,
      }),
    )
    .default([]),
  warnings: z.array(z.string()).default([]),
  summary: nullableStringToString,
  data_quality_notes: nullableStringToString,
}).strict();

export function parseExtraction(payload: unknown): ExtractionResult {
  const parsed = extractionSchema.safeParse(payload);
  if (!parsed.success) {
    return extractionSchema.parse({});
  }
  return parsed.data;
}

export function parseScore(payload: unknown): ScoreResult {
  const parsed = scoreSchema.safeParse(payload);
  if (!parsed.success) {
    return scoreSchema.parse({});
  }
  return parsed.data;
}
