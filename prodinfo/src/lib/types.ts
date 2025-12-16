export type Category = "food" | "cosmetics" | "other";

export interface ExtractionResult {
  product_name: string;
  brand: string;
  category: Category;
  ingredients: string[];
  allergens_found: string[];
  claims_on_label: string[];
  confidence: number;
  unreadable_parts: string[];
}

export interface GoodIngredient {
  name: string;
  why: string;
  impact: "low" | "med" | "high";
}

export interface ConcerningIngredient {
  name: string;
  why: string;
  risk: "low" | "med" | "high";
}

export interface AllergenItem {
  name: string;
  who_should_avoid: string;
}

export interface AdditiveItem {
  name: string;
  note: string;
}

export interface ScoreResult {
  buy_score_percent: number;
  verdict: "Buy" | "Maybe" | "Avoid" | string;
  good_ingredients: GoodIngredient[];
  concerning_ingredients: ConcerningIngredient[];
  allergens_or_irritants: AllergenItem[];
  additives_or_preservatives: AdditiveItem[];
  warnings: string[];
  summary: string;
  data_quality_notes: string;
}

export interface AnalysisRecord {
  id: string;
  created_at: string;
  device_id: string;
  image_urls: string[];
  raw_extraction: ExtractionResult;
  analysis_result: ScoreResult;
  product_name?: string | null;
  brand?: string | null;
  category?: Category | null;
  buy_score_percent?: number | null;
  verdict?: string | null;
  confidence?: number | null;
}

export interface HistoryItem {
  id: string;
  created_at: string;
  product_name: string | null;
  brand: string | null;
  buy_score_percent: number | null;
  verdict: string | null;
  category: Category | null;
}
