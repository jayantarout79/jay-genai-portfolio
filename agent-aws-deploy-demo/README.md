
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

⸻

AWS Deployment Guide (Elastic Beanstalk) — Streamlit AI Agent

This guide deploys a Streamlit app to AWS Elastic Beanstalk (EB) using the Python 3.12 AL2023 platform.

0) Prerequisites

Local tools
	•	Git
	•	Python 3.11+ (local is fine)
	•	AWS CLI
	•	Elastic Beanstalk CLI (EB CLI)

AWS account setup
	•	AWS account created
	•	MFA enabled
	•	IAM admin user created (or a user with enough permissions)
	•	Billing alarms/budget recommended

⸻

1) Install CLI tools

macOS (Homebrew)

brew install awscli
pip install --upgrade awsebcli

Verify

aws --version
eb --version


⸻

2) Configure AWS credentials (aws configure)

2.1 Create Access Keys (IAM user)

Login using your IAM user (recommended), not root.

AWS Console → IAM → Users → (your user) → Security credentials →
Access keys → Create access key → choose:
	•	“Command Line Interface (CLI)”
	•	Create
	•	Copy:
	•	AWS Access Key ID
	•	AWS Secret Access Key

Save them securely. You won’t see the Secret again.

2.2 Configure locally

Run:

aws configure

Paste:
	•	AWS Access Key ID: <PASTE>
	•	AWS Secret Access Key: <PASTE>
	•	Default region name: us-east-1 (or your preferred)
	•	Default output format: json

2.3 Verify identity

aws sts get-caller-identity

You should see an ARN like:
arn:aws:iam::<account-id>:user/<your-user>

⸻

3) Prepare the repo for Elastic Beanstalk

Elastic Beanstalk needs:
	•	a Python runtime (EB platform already provides)
	•	a requirements.txt
	•	a way to start Streamlit (Procfile recommended)

3.1 requirements.txt

Make sure your requirements.txt includes everything your app imports, e.g.
	•	streamlit
	•	openai
	•	tavily-python (or whatever you used)
	•	langchain / langgraph (if used)
	•	python-dotenv
	•	requests

(Keep versions pinned if possible.)

3.2 Procfile (very important)

Create a file named Procfile (no extension) in the project root:

web: streamlit run app.py --server.port=8080 --server.address=0.0.0.0 --server.headless=true

Why port 8080? EB expects the web process to listen on the environment port.
(8080 works well for Streamlit on EB.)

3.3 runtime.txt (optional)

If you want to hint Python runtime:

python-3.12

3.4 .gitignore (do not deploy secrets)

Ensure .env is ignored:

.env
__pycache__/
.cache/
.venv/


⸻

4) Initialize Elastic Beanstalk

From your project folder:

eb init

You’ll be prompted:
	1.	Select region → choose the same region you used in aws configure (e.g. us-east-1)
	2.	Application name → e.g. agent-aws-deploy-demo
	3.	Platform → choose Python
	4.	Platform version → choose Python 3.12 running on 64bit Amazon Linux 2023
	5.	CodeCommit → No (unless you want it)

⸻

5) Create the environment (first deployment)

5.1 Create environment

eb create agent-aws-env

If asked about load balancer:
	•	Choose Load balanced (default is fine)
	•	It may take several minutes

5.2 Check status

eb status

Wait until:
	•	Status: Ready
	•	Health: Green (or at least not Red)

5.3 View in browser

eb open

This should open your Streamlit app URL.

⸻

6) Configure environment variables (OpenAI/Tavily keys)

Never commit .env for production. Instead set EB environment variables.

Option A (Recommended): EB console

AWS Console → Elastic Beanstalk → Environments → agent-aws-env →
Configuration → Software → Environment properties:

Add:
	•	OPENAI_API_KEY = ...
	•	TAVILY_API_KEY = ...
	•	any others your app needs

Save → EB will redeploy.

Option B: EB CLI

eb setenv OPENAI_API_KEY="your_key" TAVILY_API_KEY="your_key"

Then redeploy:

eb deploy


⸻

7) Redeploy updates (after code changes)

Whenever you update code:

eb deploy

Check:

eb status
eb health
eb events


⸻

8) Troubleshooting (most common issues)

8.1 Health is Red but app loads sometimes

Check logs:

eb logs

Or fetch last 100 lines:

eb logs --all

Look for:
	•	missing package in requirements.txt
	•	wrong Streamlit command/port
	•	missing env variables

8.2 “OSError: [Errno 98] Address already in use”

Ensure Procfile uses port 8080 and address 0.0.0.0.

8.3 “ModuleNotFoundError”

Add that library to requirements.txt, then:

eb deploy

8.4 Tavily works locally but not on AWS

Usually env var missing. Confirm via EB env properties.

8.5 Timeout / slow response

Consider:
	•	reducing web tool calls
	•	using caching
	•	increasing EB instance size (later)

⸻

9) Cost control (important)

9.1 Stop the environment when not using it

To stop charges, terminate it:

eb terminate agent-aws-env

(You can recreate later.)

9.2 Delete unused application versions (optional)

Elastic Beanstalk stores versions in S3. Clean up later if needed.

⸻

10) Clean deploy checklist

Before sharing publicly:
	•	✅ .env not committed
	•	✅ secrets only in EB env vars
	•	✅ requirements.txt complete
	•	✅ Procfile correct
	•	✅ app works from EB URL
	•	✅ README includes setup + deployment guide

⸻



