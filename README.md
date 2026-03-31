# Discord Voice Bot

Phase 1 starter for a Discord bot that:

- listens for `!ask <question>`
- sends the question to a local Ollama instance
- replies in the text channel with the model response
- can optionally log each exchange to JSON

Phase 2 starter adds:

- `!join` to join your current voice channel
- `!leave` to disconnect
- `!clear` to bulk-delete recent channel messages
- automatic voice playback of each `!ask` response after the text reply

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

   For auto-restart during development, use:

   ```powershell
   python dev_runner.py
   ```

## Voice notes

- `ffmpeg` must be available on your system path.
- The bot must have permission to connect and speak in the target voice channel.
- On `!ask`, the bot will join your current voice channel automatically if needed.
- Runtime errors and voice connection retries are written to a timestamped file in `logs/`.

## Discord setup notes

- In the Discord Developer Portal, enable the **Message Content Intent** for your bot.
- Invite the bot to your server with permission to read and send messages.

## Environment variables

- `DISCORD_ECHO_TOKEN`: required
- `OLLAMA_MODEL`: optional, defaults to `llama3.2`
- `OLLAMA_BASE_URL`: optional, defaults to `http://127.0.0.1:11434`
- `ECHO_SYSTEM_PROMPT`: optional full system prompt template for Ollama. Supports `{bot_name}`, `{guild_name}`, `{channel_name}`, and `{user_name}`
- `ECHO_CHAMBER_CHANNEL_ID`: optional, only respond in that channel when set
- `ENABLE_CONVERSATION_LOGGING`: optional, defaults to `false`
- `CONVERSATION_LOG_PATH`: optional file or directory. If you give a directory, the bot creates a timestamped JSON file per run
- `TTS_AUDIO_DIR`: optional, defaults to `generated_audio`
- `RUNTIME_LOG_PATH`: optional file or directory. If you give a directory, the bot creates a timestamped log file per run
