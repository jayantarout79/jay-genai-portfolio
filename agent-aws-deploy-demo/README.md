
AI Research Agent (Local → Cloud Ready)

A simple, production-shaped AI research agent built with LangChain, Tavily, and OpenAI, exposed through a Streamlit UI.

The agent takes a topic as input, performs web-grounded research, and generates a ~500-word structured research brief — designed to be easily deployable to AWS.

⸻

What this project demonstrates

This project is intentionally minimal, but end-to-end:
	•	Tool-augmented AI agent (LLM + web search)
	•	Deterministic agent flow (tool → reasoning → synthesis)
	•	Clean separation between UI, tools, and model logic
	•	Local development first, cloud deployment next

It is built to reflect real production patterns, not just experimentation.

⸻

Architecture (high level)
	1.	User enters a research topic in the Streamlit UI
	2.	Agent uses Tavily Search to fetch relevant web information
	3.	Results are passed to an LLM (OpenAI) for synthesis
	4.	Final output is generated as a concise research brief (~500 words)
	5.	App runs locally today, deployable to AWS without code changes

⸻

Tech stack
	•	Python 3.11
	•	Streamlit – UI layer
	•	LangChain – agent + tool orchestration
	•	Tavily API – web search grounding
	•	OpenAI API – reasoning & synthesis
	•	python-dotenv – environment management

⸻

Features
	•	Topic-based research generation
	•	Web-grounded responses (not hallucination-only)
	•	Simple agent design (easy to reason about, easy to deploy)
	•	Stateless execution (cloud-friendly)
	•	Clean dependency isolation

⸻

Local setup

1. Clone the repository

git clone <your-repo-url>
cd <repo-folder>


⸻

2. Create and activate virtual environment

python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows


⸻

3. Install dependencies

pip install -r requirements.txt


⸻

4. Environment variables

Create a .env file in the project root:

OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key


⸻

5. Run locally

streamlit run app.py

Open browser at:

http://localhost:8501


⸻

Why this design

This agent is intentionally:
	•	Stateless → easy AWS deployment
	•	Tool-first → grounded outputs
	•	Minimal → fewer failure modes
	•	Readable → easy to extend later

The goal is to master deployment discipline, not just model usage.

⸻

Planned next steps
	•	Dockerize the app
	•	Deploy to AWS App Runner
	•	Add basic request logging
	•	Optional: rate limiting / auth
	•	Optional: swap Tavily for other search tools

⸻

Notes on cost & safety
	•	Web search + LLM calls incur API costs
	•	Keep prompts bounded for predictable spend
	•	Suitable for demos, learning, and controlled workloads

⸻

License

This project is for learning and demonstration purposes.

⸻


