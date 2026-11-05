import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing in your .env file.")
    return Groq(api_key=api_key)

def get_default_model(client: Groq) -> str:
    models = [
        m.id for m in client.models.list().data
        if not any(x in m.id.lower() for x in ["whisper", "tts", "guard", "embed", "vision"])
    ]
    for preferred in ["llama-3.3-70b", "llama-3.1-8b-instant", "llama3", "qwen", "mixtral", "gemma"]:
        for m in models:
            if preferred in m.lower():
                return m
    return models[0]

def generate_answer(prompt: str, model: str | None = None) -> str:
    client = get_client()
    active_model = model or get_default_model(client)
    response = client.chat.completions.create(
        model=active_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    client = get_client()
    chosen = get_default_model(client)
    print(f"Using active Groq model: {chosen}")
    print(generate_answer("Reply with: Groq is connected and ready.", model=chosen))