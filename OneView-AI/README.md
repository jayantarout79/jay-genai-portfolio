# OneView AI – Multimodal Video & Document Insights

## What it does
- Upload meeting videos, PDFs, CSVs, logs, and images in one place.
- Transcribes video audio and blends all uploads into AI-generated insights, actions, and a narration script.
- Sends the narration to D-ID to render an avatar video; shows the playable link and allows download/email sharing.

## Tech stack
- **UI:** Streamlit
- **AI:** OpenAI API (Whisper transcription + GPT for reasoning)
- **Avatar video:** D-ID Talks API
- **Data parsing:** pandas (CSV), pypdf (PDF), Pillow (image metadata)
- **Email:** SMTP (configured via `.env`)

## Project flow
1) **Upload** your video and supporting files (CSV, PDF, TXT/LOG, JPG/PNG).
2) **Transcription** runs on the video audio (OpenAI Whisper).
3) **Insight synthesis** blends transcript snippets + document summaries via GPT.
4) **Avatar render** sends the narration to D-ID and polls until the video is ready.
5) **Delivery** presents insights/actions, a playable/downloadable video, and a “Send via Email” button.

## Setup
1) Install deps: `pip install -r requirements.txt`
2) Add a `.env` with:
   - `OPENAI_API_KEY`
   - `D_ID_API_KEY` and `D_ID_SOURCE_URL` (public image URL for the avatar)
   - SMTP for email (Gmail example, use App Password):
     - `SMTP_HOST=smtp.gmail.com`
     - `SMTP_PORT=587`
     - `SMTP_USER=your-email@gmail.com`
     - `SMTP_PASS=your-app-password`
     - `SMTP_FROM=your-email@gmail.com`
     - `SMTP_TO=recipient@example.com`
3) Run: `streamlit run app.py`

## Using the app
- Click **Browse files** to add your assets, then hit **Run Multimodal Pipeline**.
- Watch the flow progress update step by step; once D-ID finishes, the video appears with download/email options.
- The email body includes the executive summary, insights, actions, and the video link.
