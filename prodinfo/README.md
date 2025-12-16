# IngrediScore Web (Next.js + Supabase + OpenAI)

iOS-style PWA that scores food/cosmetic labels. Anonymous device-only identity (no login). Each analysis uploads images to Supabase Storage, calls OpenAI vision + text for extraction + scoring, saves a row in Postgres, and shows per-device history.

## Features
- Mobile-first iOS look: large titles, glass tab bar, rounded cards, safe-area padding, centered phone frame on desktop.
- Analyze up to 2 photos (camera capture enabled) and auto-save to Supabase.
- Per-device history (device_id stored in localStorage), detail view, delete and clear-history.
- Rate limit: max 5 analyses/day per device server-side.
- PWA-ready: manifest, icons, iOS web-app meta.

## Quickstart
1) Install deps
```bash
npm install
```

2) Configure env
Copy `.env.local.example` to `.env.local` and fill:
- `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` (service role only; never expose to client)
- `SUPABASE_STORAGE_BUCKET=product-labels`
- `OPENAI_API_KEY`
- Optional: `OPENAI_VISION_MODEL`, `OPENAI_TEXT_MODEL`

3) Supabase setup
- Create a public storage bucket named `product-labels`.
- Run the SQL in `supabase.sql` to create `product_analyses` and indexes.

4) Run dev
```bash
npm run dev
```
Open http://localhost:3000. Use a phone to test camera capture on the Analyze tab.

5) Deploy
- Vercel works out of the box. Add the env vars above.
- Ensure `product-labels` bucket stays public or adjust the API to sign URLs.

## API routes
- `POST /api/analyze` (multipart) with `device_id` and `images[]` (1–2 files). Uploads to Storage, calls OpenAI (vision then text), stores row, returns JSON.
- `GET /api/history?device_id=...` last 50 rows for that device.
- `GET /api/analysis/:id?device_id=...` detail (verifies device).
- `DELETE /api/analysis/:id?device_id=...` delete one.
- `DELETE /api/history?device_id=...` clear all for device.

## Notes
- Anonymous identity only; clearing site data regenerates device_id and history will no longer match (expected).
- UI disclaimer: informational only, not medical advice.
- Tailwind v4 used via `@import "tailwindcss"` in `globals.css`.

