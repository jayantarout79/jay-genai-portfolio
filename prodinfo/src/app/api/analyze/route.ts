import { Buffer } from "buffer";
import { NextRequest, NextResponse } from "next/server";
import { openaiClient } from "@/lib/openai";
import { supabaseServerClient } from "@/lib/supabase";
import { parseExtraction, parseScore } from "@/lib/validation";
import type { ExtractionResult, ScoreResult } from "@/lib/types";

const DAILY_ANALYSIS_LIMIT = Number.parseInt(process.env.DAILY_ANALYSIS_LIMIT ?? "0", 10);
const PRIMARY_VISION_MODEL = process.env.OPENAI_VISION_MODEL_CHEAP || process.env.OPENAI_VISION_MODEL || "gpt-4o-mini";
const FALLBACK_VISION_MODEL = process.env.OPENAI_VISION_MODEL || "gpt-4o";
const PRIMARY_TEXT_MODEL = process.env.OPENAI_TEXT_MODEL_CHEAP || process.env.OPENAI_TEXT_MODEL || "gpt-4o-mini";
const FALLBACK_TEXT_MODEL = process.env.OPENAI_TEXT_MODEL || "gpt-4o";
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/heic", "image/heif"];
const MAX_SIZE = 8 * 1024 * 1024;
const OPENAI_TIMEOUT_MS = Number.parseInt(process.env.OPENAI_TIMEOUT_MS ?? "45000", 10);

export async function POST(req: NextRequest) {
  if (!supabaseServerClient || !openaiClient) {
    return NextResponse.json({ error: "Server not configured" }, { status: 500 });
  }

  const form = await req.formData();
  const deviceId = form.get("device_id");
  const extractedTextRaw = form.get("extracted_text");
  const extractedText = typeof extractedTextRaw === "string" ? extractedTextRaw.trim() : "";
  const images = form
    .getAll("images")
    .filter((f): f is File => f instanceof File && f.size > 0);

  if (!deviceId || typeof deviceId !== "string") {
    return NextResponse.json({ error: "device_id required" }, { status: 400 });
  }

  if ((!images.length && !extractedText) || images.length > 2) {
    return NextResponse.json({ error: "Provide text or 1-2 images" }, { status: 400 });
  }

  for (const file of images) {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return NextResponse.json({ error: "Only jpeg/png/heic allowed" }, { status: 400 });
    }
    if (file.size > MAX_SIZE) {
      return NextResponse.json({ error: "Each image must be < 8MB" }, { status: 400 });
    }
  }

  if (Number.isFinite(DAILY_ANALYSIS_LIMIT) && DAILY_ANALYSIS_LIMIT > 0) {
    const startOfDay = new Date();
    startOfDay.setUTCHours(0, 0, 0, 0);
    const rate = await supabaseServerClient
      .from("product_analyses")
      .select("id", { count: "exact", head: true })
      .eq("device_id", deviceId)
      .gte("created_at", startOfDay.toISOString());

    if ((rate.count ?? 0) >= DAILY_ANALYSIS_LIMIT) {
      return NextResponse.json({ error: "Daily limit reached" }, { status: 429 });
    }
  }

  let extraction: ExtractionResult | null = null;
  let scoring: ScoreResult | null = null;

  const base64Images: string[] = [];
  for (const file of images) {
    const arrayBuffer = await file.arrayBuffer();
    const mime = file.type || "image/jpeg";
    const base64 = Buffer.from(arrayBuffer).toString("base64");
    base64Images.push(`data:${mime};base64,${base64}`);
  }

  try {
    if (extractedText) {
      extraction = await runTextExtraction(extractedText);
      if (!hasIngredients(extraction) && base64Images.length) {
        extraction = await runExtractionWithFallback(base64Images);
      }
    } else if (base64Images.length) {
      extraction = await runExtractionWithFallback(base64Images);
    }

    if (!extraction) {
      return NextResponse.json({ error: "Unable to extract data" }, { status: 500 });
    }

    scoring = await runScoringWithFallback(extraction);
  } catch (error) {
    console.error("OpenAI analysis failed", error);
    const message =
      error instanceof Error
        ? error.message
        : typeof error === "string"
          ? error
          : "OpenAI analysis failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }

  const payload = {
    device_id: deviceId,
    image_urls: [],
    raw_extraction: extraction!,
    analysis_result: scoring!,
    product_name: extraction!.product_name || null,
    brand: extraction!.brand || null,
    category: extraction!.category || null,
    buy_score_percent: scoring!.buy_score_percent ?? null,
    verdict: scoring!.verdict ?? null,
    confidence: extraction!.confidence ?? null,
  };

  const insert = await supabaseServerClient
    .from("product_analyses")
    .insert(payload)
    .select()
    .single();

  if (insert.error) {
    console.error(insert.error);
    return NextResponse.json({ error: "Failed to save analysis" }, { status: 500 });
  }

  return NextResponse.json(insert.data);
}

async function runExtractionWithFallback(imageInputs: string[]): Promise<ExtractionResult> {
  const first = await runExtraction(imageInputs, PRIMARY_VISION_MODEL);
  if (hasIngredients(first) || PRIMARY_VISION_MODEL === FALLBACK_VISION_MODEL) return first;
  const second = await runExtraction(imageInputs, FALLBACK_VISION_MODEL);
  return hasIngredients(second) ? second : first;
}

async function runExtraction(imageUrls: string[], model: string): Promise<ExtractionResult> {
  if (!openaiClient) throw new Error("OpenAI not configured");

  const systemPrompt = `You are a precise OCR and ingredient parser. Avoid hallucinations; if a field is missing, return null/empty. Read the INGREDIENTS panel carefully and transcribe the full list exactly, including percentages and descriptors; do not skip items or collapse them. If an item is partially unreadable, capture the readable part and append "(unreadable)".
Return JSON only with keys:
- product_name (string, null if missing),
- brand (string, null if missing),
- category ("food"|"cosmetics"|"other"),
- ingredients (array of strings, one per ingredient; must be populated when an Ingredients section is visible, even if some items are partially unreadable),
- allergens_found (array of strings),
- claims_on_label (array of strings),
- confidence (0-1 numeric),
- unreadable_parts (array of strings describing smudged/blocked text or why ingredients were missing).`;

  const res = await openaiClient.chat.completions.create({
    model,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: systemPrompt },
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Extract product and ingredient data. Focus on the INGREDIENTS section; list every ingredient as a separate string, preserving percentages and descriptors. Do not return an empty ingredients array if an Ingredients section is visible—return the readable items and add any unreadable fragments to unreadable_parts. Respond with JSON only.",
          },
          ...imageUrls.map((url) => ({ type: "image_url" as const, image_url: { url } })),
        ],
      },
    ],
    signal: AbortSignal.timeout(OPENAI_TIMEOUT_MS),
  });

  return parseExtraction(safeJson(res.choices[0].message.content));
}

async function runTextExtraction(text: string): Promise<ExtractionResult> {
  if (!openaiClient) throw new Error("OpenAI not configured");

  const systemPrompt = `You are a precise ingredient parser. Avoid hallucinations; if a field is missing, return null/empty. Parse the provided text as if it came from an ingredient label. Respond with JSON only.`;

  const res = await openaiClient.chat.completions.create({
    model: PRIMARY_TEXT_MODEL,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: text },
    ],
    signal: AbortSignal.timeout(OPENAI_TIMEOUT_MS),
  });

  const primary = parseExtraction(safeJson(res.choices[0].message.content));
  if (hasIngredients(primary) || PRIMARY_TEXT_MODEL === FALLBACK_TEXT_MODEL) return primary;

  const fallback = await openaiClient.chat.completions.create({
    model: FALLBACK_TEXT_MODEL,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: text },
    ],
    signal: AbortSignal.timeout(OPENAI_TIMEOUT_MS),
  });

  const parsedFallback = parseExtraction(safeJson(fallback.choices[0].message.content));
  return hasIngredients(parsedFallback) ? parsedFallback : primary;
}

async function runScoringWithFallback(extraction: ExtractionResult): Promise<ScoreResult> {
  const primary = await runScoring(extraction, PRIMARY_TEXT_MODEL);
  if (hasMeaningfulScore(primary) || PRIMARY_TEXT_MODEL === FALLBACK_TEXT_MODEL) return primary;
  const fallback = await runScoring(extraction, FALLBACK_TEXT_MODEL);
  return hasMeaningfulScore(fallback) ? fallback : primary;
}

async function runScoring(extraction: ExtractionResult, model: string): Promise<ScoreResult> {
  if (!openaiClient) throw new Error("OpenAI not configured");

  const systemPrompt = `You score products based on ingredients. Be concise, avoid speculation, and penalize missing or low-confidence data.
Return JSON only with keys:
- buy_score_percent (0-100),
- verdict ("Buy"|"Maybe"|"Avoid"),
- good_ingredients [{name, why, impact: "low"|"med"|"high"}],
- concerning_ingredients [{name, why, risk: "low"|"med"|"high"}],
- allergens_or_irritants [{name, who_should_avoid}],
- additives_or_preservatives [{name, note}],
- warnings [string],
- summary (short string),
- data_quality_notes (string).`;

  const res = await openaiClient.chat.completions.create({
    model,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: systemPrompt },
      {
        role: "user",
        content: `Extraction JSON: ${JSON.stringify(extraction)}`,
      },
    ],
    signal: AbortSignal.timeout(OPENAI_TIMEOUT_MS),
  });

  const parsed = parseScore(safeJson(res.choices[0].message.content));
  if (extraction.confidence < 0.5 && parsed.buy_score_percent > 70) {
    parsed.buy_score_percent = Math.max(30, parsed.buy_score_percent - 15);
    parsed.warnings = [
      "Low OCR confidence. Recheck ingredients.",
      ...(parsed.warnings || []),
    ];
    parsed.data_quality_notes = parsed.data_quality_notes || "OCR confidence was low.";
  }
  return parsed;
}

function safeJson(input: string | null | undefined) {
  if (!input) return {};
  try {
    return JSON.parse(input);
  } catch (error) {
    console.error("Failed to parse JSON from OpenAI", error);
    return {};
  }
}

function hasIngredients(extraction: ExtractionResult | null) {
  return !!extraction && Array.isArray(extraction.ingredients) && extraction.ingredients.length > 0;
}

function hasMeaningfulScore(score: ScoreResult | null) {
  if (!score) return false;
  const hasInsights =
    (score.good_ingredients?.length ?? 0) > 0 ||
    (score.concerning_ingredients?.length ?? 0) > 0 ||
    (score.warnings?.length ?? 0) > 0;
  return hasInsights || (score.buy_score_percent ?? 0) > 0;
}
