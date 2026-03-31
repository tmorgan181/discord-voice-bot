import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import discord
import pyttsx3
import requests
from dotenv import load_dotenv

try:
    import davey  # type: ignore
except ImportError:
    davey = None


DEFAULT_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
MAX_DISCORD_MESSAGE_LEN = 1900
DEFAULT_CONVERSATION_LOG_DIR = Path("conversations")
DEFAULT_AUDIO_DIR = Path("generated_audio")
DEFAULT_RUNTIME_LOG_DIR = Path("logs")
VOICE_CONNECT_RETRIES = 3
VOICE_RETRY_DELAY_SECONDS = 2
VOICE_HEALTH_LOG_INTERVAL_SECONDS = 30
DEFAULT_CLEAR_COUNT = 25
MAX_CLEAR_COUNT = 100
SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
SESSION_RUNTIME_LOG_PATH: Path | None = None
SESSION_CONVERSATION_LOG_PATH: Path | None = None


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_required_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    joined = ", ".join(names)
    raise RuntimeError(f"Missing required environment variable. Tried: {joined}")


def resolve_runtime_log_path() -> Path:
    configured = os.getenv("RUNTIME_LOG_PATH", "").strip()
    if configured:
        configured_path = Path(configured)
        if configured_path.suffix:
            return configured_path
        return configured_path / f"bot-{SESSION_TIMESTAMP}.log"
    return DEFAULT_RUNTIME_LOG_DIR / f"bot-{SESSION_TIMESTAMP}.log"


def resolve_conversation_log_path() -> Path:
    configured = os.getenv("CONVERSATION_LOG_PATH", "").strip()
    if configured:
        configured_path = Path(configured)
        if configured_path.suffix:
            return configured_path
        return configured_path / f"conversation-{SESSION_TIMESTAMP}.json"
    return DEFAULT_CONVERSATION_LOG_DIR / f"conversation-{SESSION_TIMESTAMP}.json"


def setup_logging() -> logging.Logger:
    global SESSION_RUNTIME_LOG_PATH
    log_path = resolve_runtime_log_path()
    SESSION_RUNTIME_LOG_PATH = log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    discord.utils.setup_logging(level=logging.INFO, root=False)

    logger = logging.getLogger("echo")
    logger.info("Runtime logging initialized at %s", log_path)
    return logger


def log_voice_backend_status() -> None:
    if davey is None:
        logger.warning("python-davey is not installed; Discord voice playback will not work.")
        return
    logger.info("Voice backend ready with python-davey %s", getattr(davey, "__version__", "unknown"))


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


def build_system_prompt(message: discord.Message) -> str | None:
    template = os.getenv("ECHO_SYSTEM_PROMPT", "").strip()
    if not template:
        return None

    context = {
        "bot_name": str(client.user) if client.user else "Echo",
        "guild_name": message.guild.name if message.guild else "",
        "channel_name": getattr(message.channel, "name", ""),
        "user_name": (
            message.author.display_name
            if isinstance(message.author, discord.Member)
            else str(message.author)
        ),
    }

    try:
        return template.format(**context)
    except KeyError as exc:
        logger.warning("Unknown placeholder in ECHO_SYSTEM_PROMPT: %s", exc)
        return template


def ask_ollama_with_context(message: discord.Message, prompt: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    system_prompt = build_system_prompt(message)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system_prompt:
        payload["system"] = system_prompt

    response = requests.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()
    return data.get("response", "").strip()


def append_conversation_log(entry: dict) -> None:
    global SESSION_CONVERSATION_LOG_PATH
    if SESSION_CONVERSATION_LOG_PATH is None:
        SESSION_CONVERSATION_LOG_PATH = resolve_conversation_log_path()

    log_path = SESSION_CONVERSATION_LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if not log_path.exists():
            with log_path.open("w", encoding="utf-8") as fh:
                json.dump([entry], fh, indent=2, ensure_ascii=True)
            logger.info("Created conversation log at %s", log_path)
            return

        with log_path.open("r+", encoding="utf-8") as fh:
            try:
                loaded = json.load(fh)
                if not isinstance(loaded, list):
                    loaded = []
            except json.JSONDecodeError:
                loaded = []

            loaded.append(entry)
            fh.seek(0)
            json.dump(loaded, fh, indent=2, ensure_ascii=True)
            fh.truncate()
    except OSError:
        logger.exception("Could not write conversation log at %s", log_path)


def synthesize_tts_to_file(text: str) -> Path:
    audio_dir = Path(os.getenv("TTS_AUDIO_DIR", DEFAULT_AUDIO_DIR))
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_path = audio_dir / f"echo-{uuid.uuid4().hex}.wav"
    engine = pyttsx3.init()
    engine.save_to_file(text, str(audio_path))
    engine.runAndWait()
    engine.stop()

    if not audio_path.exists():
        raise RuntimeError("TTS audio file was not created.")

    logger.info("Generated TTS audio at %s", audio_path)
    return audio_path


async def send_error(channel: discord.abc.Messageable, text: str) -> None:
    try:
        await channel.send(text)
    except Exception:
        logger.exception("Failed to send Discord error message: %s", text)


async def handle_clear_command(message: discord.Message, content: str) -> None:
    if not message.guild:
        await send_error(message.channel, "The `!clear` command only works inside a server.")
        return

    channel = message.channel
    if not isinstance(channel, discord.abc.GuildChannel) or not hasattr(channel, "purge"):
        await send_error(
            message.channel,
            "This channel type does not support message clearing.",
        )
        return

    if not isinstance(message.author, discord.Member):
        await send_error(message.channel, "I couldn't verify your server permissions for `!clear`.")
        return

    if not message.author.guild_permissions.manage_messages:
        await send_error(message.channel, "You need the `Manage Messages` permission to use `!clear`.")
        return

    me = message.guild.me
    if not me:
        await send_error(message.channel, "I couldn't verify my permissions in this server.")
        return

    if not channel.permissions_for(me).manage_messages:
        await send_error(
            message.channel,
            "I need the `Manage Messages` permission in this channel to clear messages.",
        )
        return

    parts = content.split(maxsplit=1)
    count = DEFAULT_CLEAR_COUNT
    if len(parts) == 2:
        try:
            count = int(parts[1])
        except ValueError:
            await send_error(message.channel, "Usage: `!clear` or `!clear 25`")
            return

    if count < 1 or count > MAX_CLEAR_COUNT:
        await send_error(
            message.channel,
            f"`!clear` accepts a number between 1 and {MAX_CLEAR_COUNT}.",
        )
        return

    try:
        deleted = await channel.purge(limit=count + 1, bulk=True)
        deleted_count = max(len(deleted) - 1, 0)
        logger.info(
            "Cleared %s messages in channel %s requested by user %s",
            deleted_count,
            channel.id,
            message.author.id,
        )
        confirmation = await message.channel.send(f"Cleared {deleted_count} messages.")
        await confirmation.delete(delay=5)
    except discord.Forbidden:
        logger.exception("Missing permissions while clearing messages")
        await send_error(
            message.channel,
            "I couldn't clear messages because Discord denied the action.",
        )
    except discord.HTTPException:
        logger.exception("Discord failed while clearing messages")
        await send_error(
            message.channel,
            "Discord couldn't clear those messages. Messages older than 14 days cannot be bulk deleted.",
        )


async def reset_voice_client(guild: discord.Guild | None) -> None:
    if guild and guild.voice_client:
        try:
            logger.info(
                "Resetting stale voice client for guild=%s channel=%s connected=%s playing=%s",
                guild.id,
                guild.voice_client.channel.id if guild.voice_client.channel else None,
                guild.voice_client.is_connected(),
                guild.voice_client.is_playing(),
            )
            await guild.voice_client.disconnect(force=True)
        except Exception:
            logger.exception("Failed to disconnect stale voice client for guild %s", guild.id)


def voice_state_snapshot(voice_client: discord.VoiceClient | None) -> dict:
    if voice_client is None:
        return {
            "connected": False,
            "channel_id": None,
            "playing": False,
            "paused": False,
            "latency": None,
        }
    return {
        "connected": voice_client.is_connected(),
        "channel_id": voice_client.channel.id if voice_client.channel else None,
        "playing": voice_client.is_playing(),
        "paused": voice_client.is_paused(),
        "latency": getattr(voice_client, "latency", None),
    }


async def voice_health_monitor() -> None:
    await client.wait_until_ready()
    while not client.is_closed():
        try:
            for guild in client.guilds:
                voice_client = guild.voice_client
                if voice_client is None:
                    continue
                snapshot = voice_state_snapshot(voice_client)
                logger.info(
                    "Voice health guild=%s channel=%s connected=%s playing=%s paused=%s latency=%s",
                    guild.id,
                    snapshot["channel_id"],
                    snapshot["connected"],
                    snapshot["playing"],
                    snapshot["paused"],
                    snapshot["latency"],
                )
        except Exception:
            logger.exception("Voice health monitor failed")
        await asyncio.sleep(VOICE_HEALTH_LOG_INTERVAL_SECONDS)


async def connect_voice_with_retries(
    guild: discord.Guild, target_channel: discord.VoiceChannel
) -> discord.VoiceClient:
    last_error: Exception | None = None

    for attempt in range(1, VOICE_CONNECT_RETRIES + 1):
        try:
            voice_client = guild.voice_client
            logger.info(
                "Voice connect attempt %s/%s for guild=%s target_channel=%s existing_state=%s",
                attempt,
                VOICE_CONNECT_RETRIES,
                guild.id,
                target_channel.id,
                voice_state_snapshot(voice_client),
            )

            if voice_client and voice_client.channel and voice_client.channel.id == target_channel.id:
                if voice_client.is_connected():
                    logger.info(
                        "Reusing existing voice connection in %s for guild %s",
                        target_channel.name,
                        guild.id,
                    )
                    return voice_client
                await reset_voice_client(guild)

            if voice_client:
                logger.info(
                    "Moving existing voice client to %s (attempt %s/%s)",
                    target_channel.name,
                    attempt,
                    VOICE_CONNECT_RETRIES,
                )
                await voice_client.move_to(target_channel)
                await asyncio.sleep(1)
                if voice_client.is_connected():
                    return voice_client
                raise discord.ClientException("Voice client moved but did not become connected.")

            logger.info(
                "Connecting to voice channel %s in guild %s (attempt %s/%s)",
                target_channel.name,
                guild.id,
                attempt,
                VOICE_CONNECT_RETRIES,
            )
            new_voice_client = await target_channel.connect(
                reconnect=True,
                timeout=20.0,
                self_deaf=True,
            )
            if new_voice_client.is_connected():
                return new_voice_client
            raise discord.ClientException("Voice connection completed without a connected client.")
        except Exception as exc:
            last_error = exc
            logger.exception(
                "Voice connection attempt %s/%s failed for guild %s channel %s",
                attempt,
                VOICE_CONNECT_RETRIES,
                guild.id,
                target_channel.id,
            )
            await reset_voice_client(guild)
            if attempt < VOICE_CONNECT_RETRIES:
                await asyncio.sleep(VOICE_RETRY_DELAY_SECONDS)

    raise discord.ClientException(
        f"Unable to connect to voice after {VOICE_CONNECT_RETRIES} attempts: {last_error}"
    )


async def ensure_voice_client(message: discord.Message) -> discord.VoiceClient | None:
    if not message.guild:
        await send_error(message.channel, "Voice features only work inside a server.")
        return None

    if not isinstance(message.author, discord.Member) or not message.author.voice:
        await send_error(message.channel, "Join a voice channel first, then try again.")
        return None

    target_channel = message.author.voice.channel
    logger.info(
        "Ensuring voice client for guild=%s channel=%s user=%s",
        message.guild.id,
        target_channel.id,
        message.author.id,
    )
    return await connect_voice_with_retries(message.guild, target_channel)


async def speak_text(voice_client: discord.VoiceClient, text: str) -> None:
    logger.info(
        "Preparing TTS playback channel=%s chars=%s state=%s",
        voice_client.channel.id if voice_client.channel else None,
        len(text),
        voice_state_snapshot(voice_client),
    )
    audio_path = await asyncio.to_thread(synthesize_tts_to_file, text)

    def cleanup(error: Exception | None) -> None:
        if error:
            logger.exception("Voice playback error", exc_info=error)
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not delete temporary audio file %s", audio_path)

    if voice_client.is_playing():
        voice_client.stop()

    source = discord.FFmpegPCMAudio(str(audio_path))
    voice_client.play(source, after=cleanup)
    logger.info("Started voice playback in channel %s", voice_client.channel.id)


load_dotenv()
logger = setup_logging()
log_voice_backend_status()

token = get_required_env("DISCORD_ECHO_TOKEN", "DISCORD_BOT_TOKEN")
allowed_channel_id = os.getenv("ECHO_CHAMBER_CHANNEL_ID")
if allowed_channel_id:
    allowed_channel_id = int(allowed_channel_id)
conversation_logging_enabled = env_flag("ENABLE_CONVERSATION_LOGGING")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True

client = discord.Client(intents=intents)
health_monitor_task: asyncio.Task | None = None


@client.event
async def on_ready() -> None:
    global health_monitor_task
    logger.info("Logged in as %s (id=%s)", client.user, client.user.id if client.user else None)
    if health_monitor_task is None or health_monitor_task.done():
        health_monitor_task = asyncio.create_task(voice_health_monitor())
        logger.info("Started voice health monitor task")


@client.event
async def on_disconnect() -> None:
    logger.warning("Discord gateway disconnected")


@client.event
async def on_resumed() -> None:
    logger.info("Discord gateway session resumed")


@client.event
async def on_error(event_method: str, *args, **kwargs) -> None:
    logger.exception("Unhandled Discord event error in %s", event_method)


@client.event
async def on_voice_state_update(
    member: discord.Member, before: discord.VoiceState, after: discord.VoiceState
) -> None:
    if not client.user or member.id != client.user.id:
        return
    logger.info(
        "Bot voice state changed guild=%s before_channel=%s after_channel=%s self_mute=%s self_deaf=%s",
        member.guild.id,
        before.channel.id if before.channel else None,
        after.channel.id if after.channel else None,
        after.self_mute,
        after.self_deaf,
    )


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author == client.user:
        return

    if allowed_channel_id is not None and message.channel.id != allowed_channel_id:
        return

    content = message.content.strip()
    parts = content.split(maxsplit=1)
    command = parts[0].lower() if parts else ""
    logger.info(
        "Received message guild=%s channel=%s user=%s content=%s",
        message.guild.id if message.guild else None,
        message.channel.id,
        message.author.id,
        content[:200],
    )

    if command == "!join":
        try:
            voice_client = await ensure_voice_client(message)
        except discord.ClientException:
            logger.exception("Discord voice join failed")
            await send_error(
                message.channel,
                "I couldn't join the voice channel after several retries. Check my Connect/Speak permissions and try again.",
            )
            return

        if voice_client is not None:
            await message.channel.send(f"Joined `{voice_client.channel.name}`.")
        return

    if command == "!leave":
        if message.guild and message.guild.voice_client:
            logger.info(
                "Manual voice disconnect requested guild=%s state=%s",
                message.guild.id,
                voice_state_snapshot(message.guild.voice_client),
            )
            await message.guild.voice_client.disconnect(force=True)
            logger.info("Disconnected from voice in guild %s", message.guild.id)
            await message.channel.send("Left the voice channel.")
        else:
            await message.channel.send("I'm not in a voice channel right now.")
        return

    if command == "!clear":
        await handle_clear_command(message, content)
        return

    if command != "!ask":
        return

    question = content[4:].strip()
    if not question:
        await message.channel.send("Usage: `!ask your question here`")
        return

    async with message.channel.typing():
        try:
            answer = await asyncio.to_thread(ask_ollama_with_context, message, question)
            logger.info("Ollama answered successfully for user %s", message.author.id)
        except requests.RequestException:
            logger.exception("Ollama request failed")
            await send_error(
                message.channel,
                "I couldn't reach Ollama. Make sure it is running locally, then try again.",
            )
            return
        except Exception:
            logger.exception("Unexpected error while generating a reply")
            await send_error(message.channel, "Something went wrong while generating a reply.")
            return

    if not answer:
        answer = "Ollama returned an empty response."

    if conversation_logging_enabled:
        append_conversation_log(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "guild_id": message.guild.id if message.guild else None,
                "guild_name": message.guild.name if message.guild else None,
                "channel_id": message.channel.id,
                "channel_name": getattr(message.channel, "name", None),
                "user_id": message.author.id,
                "username": str(message.author),
                "prompt": question,
                "response": answer,
                "model": os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
            }
        )

    for start in range(0, len(answer), MAX_DISCORD_MESSAGE_LEN):
        chunk = answer[start : start + MAX_DISCORD_MESSAGE_LEN]
        await message.channel.send(chunk)

    try:
        voice_client = await ensure_voice_client(message)
        if voice_client is not None:
            await speak_text(voice_client, answer)
    except discord.ClientException:
        logger.exception("Discord voice error after text reply")
        await send_error(
            message.channel,
            "I answered in text, but I couldn't connect to voice. Check my voice permissions and try `!join` again.",
        )
    except Exception:
        logger.exception("TTS or playback error after text reply")
        await send_error(
            message.channel,
            "I answered in text, but speaking the reply failed. Check the runtime log for details.",
        )


if __name__ == "__main__":
    client.run(token, log_handler=None)
