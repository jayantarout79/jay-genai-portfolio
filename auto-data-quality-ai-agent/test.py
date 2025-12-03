import os
import os
from dotenv import load_dotenv

load_dotenv()  # ensures .env is read from project root

url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY") or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
print("URL set:", bool(url), "Key set:", bool(key))

from supabase import create_client, Client

url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY") or os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")

print("URL set:", bool(url), "Key set:", bool(key))
if not url or not key:
    raise SystemExit("Missing URL or key in env")

print("Key prefix:", key[:8], "...len:", len(key))

try:
    client: Client = create_client(url, key)
    # Simple metadata call to prove auth works (no data needed)
    res = client.table("dq_runs").select("id").limit(1).execute()
    print("Client OK. dq_runs rows:", len(getattr(res, "data", []) or []))
except Exception as e:
    print("Client error:", e)


