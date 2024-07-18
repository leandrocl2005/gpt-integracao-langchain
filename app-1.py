from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Que cor é o céu?"}]
)

print(response)
