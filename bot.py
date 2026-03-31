import os

import discord
import requests
from dotenv import load_dotenv


DEFAULT_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
MAX_DISCORD_MESSAGE_LEN = 1900


def get_required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    joined = ", ".join(names)
    raise RuntimeError(f"Missing required environment variable. Tried: {joined}")


def ask_ollama(prompt: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    response = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


load_dotenv()
token = get_required_env("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready() -> None:
    print(f"Logged in as {client.user} (id={client.user.id})")


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author == client.user:
        return

    content = message.content.strip()
    if not content.startswith("!ask"):
        return

    question = content[4:].strip()
    if not question:
        await message.channel.send("Usage: `!ask your question here`")
        return

    async with message.channel.typing():
        try:
            answer = ask_ollama(question)
        except requests.RequestException:
            await message.channel.send(
                "I couldn't reach Ollama. Make sure it is running locally, then try again."
            )
            return
        except Exception:
            await message.channel.send("Something went wrong while generating a reply.")
            return

    if not answer:
        answer = "Ollama returned an empty response."

    for start in range(0, len(answer), MAX_DISCORD_MESSAGE_LEN):
        chunk = answer[start : start + MAX_DISCORD_MESSAGE_LEN]
        await message.channel.send(chunk)


if __name__ == "__main__":
    client.run(token)
