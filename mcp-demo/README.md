
Local LLM Agent Lab (Ollama + Streamlit)

This repository documents a small hands-on experiment where I ran a modern LLM entirely locally and wired it to a lightweight Streamlit agent.

The goal was not to build a production chatbot —
but to understand where local inference fits, where it breaks, and how to decide between local vs cloud AI in real systems.

⸻

Why this experiment?

Most AI demos assume:
	•	always-on cloud inference
	•	external APIs
	•	variable latency and cost

But many real workflows don’t need that.

This lab explores:
	•	zero-cost inference (after hardware)
	•	predictable latency
	•	data never leaving the machine
	•	fast iteration without API limits

⸻

What’s inside

1️⃣ Local LLM runtime (Ollama)
	•	Models run fully on local hardware (Apple Silicon supported)
	•	No API keys
	•	No usage-based billing
	•	HTTP interface for programmatic access

Model used in this lab:
	•	qwen2.5:7b (strong reasoning + coding balance)

⸻

2️⃣ Streamlit Agent (Minimal)

A simple agent-style interface where you can:
	•	ask questions
	•	get responses from the local model
	•	iterate quickly on prompts

This is intentionally thin:
	•	no memory store
	•	no RAG
	•	no orchestration framework

The focus is inference behavior, not tooling complexity.

⸻

3️⃣ Python integration

The agent communicates with the local model via Ollama’s HTTP API.

Key characteristics:
	•	synchronous inference
	•	no retries / no streaming
	•	deterministic behavior for experimentation

This keeps the mental model simple.

⸻

Architecture (conceptual)

User
  ↓
Streamlit UI
  ↓
Python Agent Layer
  ↓
Local Ollama Runtime
  ↓
LLM Model (on-device)

No cloud calls.
No external dependencies once installed.

⸻

What this setup is good for

✔️ Learning and experimentation
✔️ Internal tools
✔️ Prototyping agents
✔️ Data-sensitive workflows
✔️ Offline or constrained environments
✔️ Understanding latency vs cost trade-offs

⸻

Where local LLMs break down

This lab also made the limits very clear:
	•	❌ Not ideal for large context windows
	•	❌ Slower than cloud at scale
	•	❌ Hardware-bound performance
	•	❌ No built-in safety, moderation, or observability
	•	❌ Not suitable for high-concurrency workloads

Local inference is a tool, not a replacement.

⸻

How I decide: local vs cloud

I use a simple rule of thumb:

Go local when:
	•	cost sensitivity matters
	•	data must stay local
	•	concurrency is low
	•	workflows are internal or async

Go cloud when:
	•	scale matters
	•	latency SLAs are tight
	•	multimodal or long-context is required
	•	reliability > experimentation

Most real systems will use both.

⸻

How to run this locally

1️⃣ Install Ollama

brew install ollama

Start the server:

ollama serve

Pull a model:

ollama pull qwen2.5:7b


⸻

2️⃣ Install Python dependencies

pip install streamlit requests


⸻

3️⃣ Run the Streamlit agent

streamlit run app.py


⸻

Notes
	•	This repo is intentionally small.
	•	The value is in understanding system behavior, not feature count.
	•	The code is meant to be read, modified, and extended.

⸻

Next possible extensions
	•	Add prompt caching
	•	Add local embeddings
	•	Add lightweight memory
	•	Compare latency vs cloud APIs
	•	Introduce fallback routing (local → cloud)

⸻

Final thought

Modern AI engineering isn’t about picking one stack.

It’s about knowing when to use which capability, and why.

This repo captures one small but important piece of that decision space.

⸻
Additional Commands 

brew services start ollama
ollama started
brew services stop ollama

ollama run qwen2.5:7b


ollama serve > ollama.log 2>&1 &
ps aux | grep ollama
pkill ollama


curl http://127.0.0.1:11434/api/tags
