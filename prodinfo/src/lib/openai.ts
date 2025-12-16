import OpenAI from "openai";

const key = process.env.OPENAI_API_KEY;

if (!key) {
  console.warn("OPENAI_API_KEY missing. Analyze route will fail until set.");
}

export const openaiClient = key ? new OpenAI({ apiKey: key }) : undefined;
