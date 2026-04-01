# Discord Voice Bot

Phase 1 starter for a Discord bot that:

- listens for `!say <message>`
- sends the question to a local Ollama instance
- replies in the text channel with the model response
- can optionally log each exchange to JSON

Phase 2 starter adds:

- `!join` to join your current voice channel
- `!leave` to disconnect
- `!clear` to bulk-delete recent channel messages
- automatic voice playback of each `!say` response after the text reply
- hands-free listening with `!talk`, with utterances finalized after silence

Phase 3 starter adds:

- `!record [seconds]` to capture voice-channel audio and transcribe it with Whisper
- `!talk` to start hands-free listening in voice until you stop it
- `!talk [seconds]` to do a single timed voice turn, then stop
- optional reaction control on a pinned message to start on react and stop on unreact

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
- On `!say`, the bot will join your current voice channel automatically if needed.
- `!join` only joins voice.
- `!talk` starts hands-free listening, and Echo will submit an utterance after roughly 1.5 seconds of silence.
- `!stop` stops either a timed recording or the hands-free listener.
- Runtime errors and voice connection retries are written to a timestamped file in `logs/`.
- `!record` captures speech from the current voice channel for up to 30 seconds using `SpeechRecognitionSink`.
- The receive stack is still experimental upstream, so live voice recording may degrade or corrupt audio even on the latest published packages.

## Discord setup notes

- In the Discord Developer Portal, enable the **Message Content Intent** for your bot.
- Invite the bot to your server with permission to read and send messages.

## Environment variables

- `DISCORD_BOT_TOKEN`: required
- `OLLAMA_MODEL`: optional, defaults to `llama3.2`
- `OLLAMA_BASE_URL`: optional, defaults to `http://127.0.0.1:11434`
- `BOT_SYSTEM_PROMPT`: optional full system prompt template for Ollama. Supports `{bot_name}`, `{guild_name}`, `{channel_name}`, and `{user_name}`
- `BOT_HISTORY_MAX_TURNS`: optional number of recent user/assistant turns to keep in session memory after the bot joins voice, defaults to `12`
- `BOT_TEXT_CHANNEL_ID`: optional text channel ID for commands, transcripts, and replies
- `BOT_VOICE_CHANNEL_ID`: optional voice channel ID for join/talk features
- `BOT_REACTION_MESSAGE_ID`: optional message ID to use for reaction-based voice control
- `BOT_REACTION_EMOJI`: optional control emoji for the reaction trigger, defaults to `🎙️`
- `BOT_REACTION_RECORD_SECONDS`: optional max runtime for reaction-based recording before timeout, defaults to `600`
- `BOT_REACTION_STOP_DELAY_MS`: optional delay before stopping on reaction removal, defaults to `750`
- `BOT_AUTO_LISTEN_SILENCE_SECONDS`: optional silence threshold for hands-free utterance submission, defaults to `1.5`
- `BOT_AUTO_LISTEN_PHRASE_LIMIT_SECONDS`: optional max length of a single hands-free utterance before forced finalization, defaults to `30`
- `ENABLE_CONVERSATION_LOGGING`: optional, defaults to `false`
- `CONVERSATION_LOG_PATH`: optional file or directory. If you give a directory, the bot creates a timestamped JSON file per run
- `RECORDINGS_DIR`: optional directory for saved voice recordings, defaults to `recordings`
- `WHISPER_MODEL`: optional, defaults to `base.en`
- `WHISPER_AUDIO_GAIN`: optional extra gain multiplier applied before transcription, defaults to `1.15`
- `WHISPER_TARGET_RMS`: optional loudness target for adaptive normalization, defaults to `14000`
- `WHISPER_MAX_GAIN`: optional hard cap for adaptive gain, defaults to `6.0`
- `WHISPER_BEAM_SIZE`: optional Whisper decode beam size, defaults to `1`
- `TTS_AUDIO_DIR`: optional, defaults to `generated_audio`
- `RUNTIME_LOG_PATH`: optional file or directory. If you give a directory, the bot creates a timestamped log file per run

Backward-compatible aliases still work:

- `DISCORD_ECHO_TOKEN`
- `ECHO_SYSTEM_PROMPT`
- `BOT_ALLOWED_CHANNEL_ID`
- `ECHO_CHAMBER_CHANNEL_ID`
