# Discord Voice Bot

Phase 1 starter for a Discord bot that:

- listens for `!ask <question>`
- sends the question to a local Ollama instance
- replies in the text channel with the model response

## Quick start

1. Create a virtual environment if you want one:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your Discord bot token.

4. Make sure Ollama is running locally and a model is available:

   ```powershell
   ollama serve
   ollama run llama3.2
   ```

5. Start the bot:

   ```powershell
   python bot.py
   ```

## Discord setup notes

- In the Discord Developer Portal, enable the **Message Content Intent** for your bot.
- Invite the bot to your server with permission to read and send messages.

## Environment variables

- `DISCORD_BOT_TOKEN`: required
- `OLLAMA_MODEL`: optional, defaults to `llama3.2`
- `OLLAMA_BASE_URL`: optional, defaults to `http://127.0.0.1:11434`
