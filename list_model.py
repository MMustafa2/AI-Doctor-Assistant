import os
import requests
from dotenv import load_dotenv

load_dotenv()
r = requests.get("https://api.groq.com/openai/v1/models",
                 headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}, timeout=20)
r.raise_for_status()
for m in sorted(x["id"] for x in r.json()["data"]):
    print(m)