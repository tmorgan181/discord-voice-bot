import asyncio
import array
import io
import json
import logging
import math
import os
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import discord
from discord.opus import OpusError
import pyttsx3
import requests
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from discord.ext import voice_recv
from discord.ext.voice_recv import opus as voice_recv_opus
from discord.ext.voice_recv.extras.speechrecognition import SpeechRecognitionSink

try:
    import davey  # type: ignore
except ImportError:
    davey = None


DEFAULT_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
MAX_DISCORD_MESSAGE_LEN = 1900
DEFAULT_CONVERSATION_LOG_DIR = Path("conversations")
DEFAULT_AUDIO_DIR = Path("generated_audio")
DEFAULT_RECORDINGS_DIR = Path("recordings")
DEFAULT_RUNTIME_LOG_DIR = Path("logs")
VOICE_CONNECT_RETRIES = 3
VOICE_RETRY_DELAY_SECONDS = 2
VOICE_HEALTH_LOG_INTERVAL_SECONDS = 30
DEFAULT_RECORD_SECONDS = 10
MAX_RECORD_SECONDS = 30
DEFAULT_REACTION_RECORD_SECONDS = 600
DEFAULT_AUDIO_GAIN = 3.0
FFMPEG_SILENCE_INPUT = "anullsrc=r=48000:cl=stereo"
SESSION_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
SESSION_RUNTIME_LOG_PATH: Path | None = None
SESSION_CONVERSATION_LOG_PATH: Path | None = None
WHISPER_MODEL_INSTANCE: WhisperModel | None = None
ACTIVE_RECORDINGS: dict[int, dict] = {}
VOICE_RECV_PATCHED = False
ACTIVE_RECORDING_SSRCS: set[int] = set()
CORRUPT_PACKET_COUNTS: dict[int, int] = {}
REACTION_RECORDING_TASKS: dict[int, asyncio.Task] = {}


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


def get_env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


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


def log_phase3_backend_status() -> None:
    logger.info(
        "Phase 3 dependencies ready: voice_recv=%s faster_whisper=%s speech_sink=%s",
        getattr(voice_recv, "__version__", "unknown"),
        getattr(WhisperModel, "__module__", "faster_whisper"),
        SpeechRecognitionSink.__name__,
    )


def patch_voice_recv_decoder() -> None:
    global VOICE_RECV_PATCHED
    if VOICE_RECV_PATCHED:
        return

    original_decode_packet = voice_recv_opus.PacketDecoder._decode_packet
    silence_frame = b"\x00" * voice_recv_opus.Decoder.SAMPLE_SIZE * voice_recv_opus.Decoder.SAMPLES_PER_FRAME

    def safe_decode_packet(self, packet):
        try:
            return original_decode_packet(self, packet)
        except OpusError as exc:
            ssrc = getattr(packet, "ssrc", None)
            if ssrc in ACTIVE_RECORDING_SSRCS:
                CORRUPT_PACKET_COUNTS[ssrc] = CORRUPT_PACKET_COUNTS.get(ssrc, 0) + 1
            else:
                logger.warning(
                    "Ignoring corrupt Opus packet outside active recording ssrc=%s sequence=%s error=%s",
                    ssrc,
                    getattr(packet, "sequence", None),
                    exc,
                )
            return packet, silence_frame

    voice_recv_opus.PacketDecoder._decode_packet = safe_decode_packet
    VOICE_RECV_PATCHED = True
    logger.info("Patched voice receive decoder to tolerate corrupt Opus packets")


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
    template = get_env_value("BOT_SYSTEM_PROMPT", "ECHO_SYSTEM_PROMPT", default="").strip()
    if not template:
        return None

    context = {
        "bot_name": client.user.name if client.user else "Bot",
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
        logger.warning("Unknown placeholder in BOT_SYSTEM_PROMPT/ECHO_SYSTEM_PROMPT: %s", exc)
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


def log_conversation_exchange(
    message: discord.Message,
    prompt: str,
    response: str,
    source: str = "text",
) -> None:
    if not conversation_logging_enabled:
        return

    append_conversation_log(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "guild_id": message.guild.id if message.guild else None,
            "guild_name": message.guild.name if message.guild else None,
            "channel_id": message.channel.id,
            "channel_name": getattr(message.channel, "name", None),
            "user_id": message.author.id,
            "username": str(message.author),
            "display_name": (
                message.author.display_name
                if isinstance(message.author, discord.Member)
                else str(message.author)
            ),
            "prompt": prompt,
            "response": response,
            "model": os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
            "source": source,
        }
    )


async def respond_with_ollama(
    message: discord.Message,
    prompt: str,
    *,
    send_prefix: str | None = None,
    speak_reply: bool = True,
    log_source: str = "text",
) -> str | None:
    async with message.channel.typing():
        try:
            answer = await asyncio.to_thread(ask_ollama_with_context, message, prompt)
            logger.info("Ollama answered successfully for user %s source=%s", message.author.id, log_source)
        except requests.RequestException:
            logger.exception("Ollama request failed")
            await send_error(
                message.channel,
                "I couldn't reach Ollama. Make sure it is running locally, then try again.",
            )
            return None
        except Exception:
            logger.exception("Unexpected Ollama error")
            await send_error(
                message.channel,
                "Something went wrong while asking Ollama.",
            )
            return None

    log_conversation_exchange(message, prompt, answer, source=log_source)

    reply_text = f"{send_prefix}{answer}" if send_prefix else answer
    await message.channel.send(reply_text)

    if not speak_reply:
        return answer

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
            "I answered in text, but voice playback failed.",
        )

    return answer


async def handle_record_or_talk(
    context,
    duration_seconds: int,
    *,
    mode: str,
) -> None:
    try:
        voice_client = await ensure_voice_client(context)
        if voice_client is None:
            return
        logger.info(
            "Voice capture accepted mode=%s guild=%s channel=%s user=%s duration=%s",
            mode,
            context.guild.id if context.guild else None,
            context.channel.id,
            context.author.id,
            duration_seconds,
        )
        if context.guild and context.guild.id in ACTIVE_RECORDINGS:
            await send_error(
                context.channel,
                "A recording is already in progress. Use `!stop` first if you want to end it early.",
            )
            return
        status_text = "Recording and listening" if mode == "talk" else "Recording"
        await context.channel.send(f"{status_text} for {duration_seconds} seconds...")
        async with context.channel.typing():
            transcript = await run_speech_recognition_capture(
                voice_client,
                duration_seconds,
                context.guild.id,
                context.author,
            )
    except discord.ClientException:
        logger.exception("Discord voice error during recording")
        await send_error(
            context.channel,
            "I couldn't access the voice channel for recording. Try `!join` again first.",
        )
        return
    except RuntimeError as exc:
        logger.exception("Speech recognition runtime check failed")
        await send_error(
            context.channel,
            f"Recording or speech recognition failed: {exc}",
        )
        return
    except Exception:
        logger.exception("Recording or speech recognition failed")
        await send_error(
            context.channel,
            "Recording or speech recognition failed. Check the runtime log for details.",
        )
        return

    if not transcript:
        logger.info("Speech recognition completed with no detectable speech")
        transcript = "[No speech detected]"
    else:
        logger.info(
            "Speech recognition completed successfully chars=%s",
            len(transcript),
        )

    if mode == "record":
        await context.channel.send(f"Transcript: {transcript}")
        return

    if transcript == "[No speech detected]":
        await context.channel.send(f"Transcript: {transcript}")
        return

    await context.channel.send(f"Heard: {transcript}")
    await respond_with_ollama(
        context,
        transcript,
        send_prefix=None,
        speak_reply=True,
        log_source="voice",
    )


def synthesize_tts_to_file(text: str) -> Path:
    audio_dir = Path(os.getenv("TTS_AUDIO_DIR", DEFAULT_AUDIO_DIR))
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_path = audio_dir / f"tts-{uuid.uuid4().hex}.wav"
    engine = pyttsx3.init()
    engine.save_to_file(text, str(audio_path))
    engine.runAndWait()
    engine.stop()

    if not audio_path.exists():
        raise RuntimeError("TTS audio file was not created.")

    logger.info("Generated TTS audio at %s", audio_path)
    return audio_path


def get_recordings_dir() -> Path:
    configured = os.getenv("RECORDINGS_DIR", "").strip()
    if configured:
        return Path(configured)
    return DEFAULT_RECORDINGS_DIR


def create_recording_path(target_user: discord.abc.User) -> Path:
    recordings_dir = get_recordings_dir()
    recordings_dir.mkdir(parents=True, exist_ok=True)
    safe_user = str(target_user).replace(" ", "_").replace("#", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return recordings_dir / f"recording-{timestamp}-{safe_user}.wav"


def append_wav_frames(audio_bytes: bytes, pcm_chunks: list[bytes]) -> tuple[wave._wave_params, bytes]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    pcm_chunks.append(frames)
    return params, frames


def save_recording_wav(path: Path, params: wave._wave_params | None, pcm_chunks: list[bytes]) -> bool:
    if params is None or not pcm_chunks:
        return False

    with wave.open(str(path), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(b"".join(pcm_chunks))
    return True


def compute_pcm_rms(frames: bytes, sample_width: int) -> float:
    if not frames:
        return 0.0
    if sample_width != 2:
        return 0.0

    samples = array.array("h")
    samples.frombytes(frames)
    if not samples:
        return 0.0

    square_sum = sum(sample * sample for sample in samples)
    return math.sqrt(square_sum / len(samples))


def scale_pcm_frames(frames: bytes, sample_width: int, gain: float) -> bytes:
    if not frames or gain == 1.0:
        return frames
    if sample_width != 2:
        return frames

    samples = array.array("h")
    samples.frombytes(frames)
    for index, sample in enumerate(samples):
        scaled = int(sample * gain)
        if scaled > 32767:
            scaled = 32767
        elif scaled < -32768:
            scaled = -32768
        samples[index] = scaled
    return samples.tobytes()


def amplify_wav_bytes(audio_bytes: bytes) -> tuple[bytes, float]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())

    rms = compute_pcm_rms(frames, params.sampwidth)
    gain = float(os.getenv("WHISPER_AUDIO_GAIN", str(DEFAULT_AUDIO_GAIN)))
    boosted_frames = scale_pcm_frames(frames, params.sampwidth, gain)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setparams(params)
        writer.writeframes(boosted_frames)
    return buffer.getvalue(), float(rms)


def get_whisper_model() -> WhisperModel:
    global WHISPER_MODEL_INSTANCE
    if WHISPER_MODEL_INSTANCE is None:
        model_name = os.getenv("WHISPER_MODEL", "tiny.en")
        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        logger.info(
            "Loading Whisper model name=%s device=%s compute_type=%s",
            model_name,
            device,
            compute_type,
        )
        WHISPER_MODEL_INSTANCE = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
    return WHISPER_MODEL_INSTANCE


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
    count: int | None = None
    if len(parts) == 2:
        try:
            count = int(parts[1])
        except ValueError:
            await send_error(message.channel, "Usage: `!clear` or `!clear 50`")
            return

        if count < 1:
            await send_error(
                message.channel,
                "`!clear` only accepts a positive number.",
            )
            return

    try:
        purge_limit = None if count is None else count + 1
        deleted = await channel.purge(
            limit=purge_limit,
            bulk=True,
            check=lambda msg: not msg.pinned,
        )
        deleted_count = max(len(deleted) - 1, 0)
        logger.info(
            "Cleared %s messages in channel %s requested by user %s limit=%s",
            deleted_count,
            channel.id,
            message.author.id,
            count,
        )
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


def get_current_voice_client(guild: discord.Guild | None) -> discord.VoiceClient | None:
    if guild is None:
        return None
    if guild.voice_client is not None:
        return guild.voice_client

    for voice_client in client.voice_clients:
        if voice_client.guild and voice_client.guild.id == guild.id:
            return voice_client
    return None


def is_idle_keepalive_active(voice_client: discord.VoiceClient | None) -> bool:
    return bool(voice_client and getattr(voice_client, "_idle_keepalive_active", False))


def start_idle_keepalive(voice_client: discord.VoiceClient) -> None:
    if not voice_client.is_connected():
        logger.info("Skipping idle keepalive because voice client is not connected")
        return
    if voice_client.is_playing():
        logger.info(
            "Skipping idle keepalive because audio is already playing in channel %s",
            voice_client.channel.id if voice_client.channel else None,
        )
        return

    source = discord.FFmpegPCMAudio(
        source=FFMPEG_SILENCE_INPUT,
        before_options="-f lavfi",
        options="-f s16le -ar 48000 -ac 2",
    )
    setattr(voice_client, "_idle_keepalive_active", True)
    voice_client.play(source)
    logger.info(
        "Started silent keepalive playback in channel %s",
        voice_client.channel.id if voice_client.channel else None,
    )


def stop_idle_keepalive(voice_client: discord.VoiceClient) -> None:
    if is_idle_keepalive_active(voice_client):
        logger.info(
            "Stopping silent keepalive playback in channel %s",
            voice_client.channel.id if voice_client.channel else None,
        )
        setattr(voice_client, "_idle_keepalive_active", False)
        voice_client.stop()


def stop_voice_capture(voice_client: discord.VoiceClient) -> None:
    if isinstance(voice_client, voice_recv.VoiceRecvClient) and voice_client.is_listening():
        logger.info(
            "Stopping active voice capture in channel %s",
            voice_client.channel.id if voice_client.channel else None,
        )
        voice_client.stop_listening()


def is_recoverable_capture_error(error: object) -> bool:
    if error is None:
        return False
    text = str(error).strip().lower()
    return "corrupted stream" in text


def stop_active_recording(guild_id: int) -> bool:
    session = ACTIVE_RECORDINGS.get(guild_id)
    if not session:
        return False

    stop_event = session.get("stop_event")
    voice_client = session.get("voice_client")
    logger.info("Stop requested for active recording in guild=%s", guild_id)

    if stop_event is not None and not stop_event.is_set():
        stop_event.set()

    if isinstance(voice_client, discord.VoiceClient):
        stop_voice_capture(voice_client)

    return True


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
                    "Voice health guild=%s channel=%s connected=%s playing=%s paused=%s latency=%s idle_keepalive=%s",
                    guild.id,
                    snapshot["channel_id"],
                    snapshot["connected"],
                    snapshot["playing"],
                    snapshot["paused"],
                    snapshot["latency"],
                    is_idle_keepalive_active(voice_client),
                )
        except Exception:
            logger.exception("Voice health monitor failed")
        await asyncio.sleep(VOICE_HEALTH_LOG_INTERVAL_SECONDS)


async def connect_voice_with_retries(
    guild: discord.Guild, target_channel: discord.VoiceChannel
) -> voice_recv.VoiceRecvClient:
    last_error: Exception | None = None

    for attempt in range(1, VOICE_CONNECT_RETRIES + 1):
        try:
            voice_client = get_current_voice_client(guild)
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
                    if not isinstance(voice_client, voice_recv.VoiceRecvClient):
                        logger.info("Existing voice client is not VoiceRecvClient; resetting it")
                        await reset_voice_client(guild)
                    else:
                        logger.info(
                            "Reusing existing voice connection in %s for guild %s",
                            target_channel.name,
                            guild.id,
                        )
                        start_idle_keepalive(voice_client)
                        return voice_client
                if get_current_voice_client(guild):
                    await reset_voice_client(guild)
                voice_client = get_current_voice_client(guild)

            if voice_client:
                if not isinstance(voice_client, voice_recv.VoiceRecvClient):
                    logger.info("Resetting non-receive voice client before move/connect")
                    await reset_voice_client(guild)
                    voice_client = get_current_voice_client(guild)

            if voice_client and voice_client.channel and voice_client.channel.id == target_channel.id:
                if voice_client.is_connected():
                    logger.info(
                        "Reusing existing voice connection in %s for guild %s",
                        target_channel.name,
                        guild.id,
                    )
                    start_idle_keepalive(voice_client)
                    return voice_client

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
                    start_idle_keepalive(voice_client)
                    return voice_client
                raise discord.ClientException("Voice client moved but did not become connected.")

            logger.info(
                "Connecting to voice channel %s in guild %s (attempt %s/%s, self_deaf=%s)",
                target_channel.name,
                guild.id,
                attempt,
                VOICE_CONNECT_RETRIES,
                False,
            )
            new_voice_client = await target_channel.connect(
                reconnect=True,
                timeout=20.0,
                self_deaf=False,
                cls=voice_recv.VoiceRecvClient,
            )
            if new_voice_client.is_connected():
                start_idle_keepalive(new_voice_client)
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


async def ensure_voice_client(message: discord.Message) -> voice_recv.VoiceRecvClient | None:
    if not message.guild:
        await send_error(message.channel, "Voice features only work inside a server.")
        return None

    configured_voice_channel_id = get_env_value("BOT_VOICE_CHANNEL_ID")
    target_channel: discord.VoiceChannel | None = None

    if configured_voice_channel_id:
        channel = message.guild.get_channel(int(configured_voice_channel_id))
        if not isinstance(channel, discord.VoiceChannel):
            await send_error(
                message.channel,
                "The configured `BOT_VOICE_CHANNEL_ID` is missing or is not a voice channel.",
            )
            return None
        target_channel = channel
    else:
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
    voice_client = await connect_voice_with_retries(message.guild, target_channel)
    await ensure_control_message_reaction(message.channel)
    return voice_client


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
        setattr(voice_client, "_idle_keepalive_active", False)
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not delete temporary audio file %s", audio_path)
        if voice_client.is_connected():
            try:
                start_idle_keepalive(voice_client)
            except Exception:
                logger.exception("Failed to restart idle keepalive after TTS playback")

    if voice_client.is_playing():
        stop_idle_keepalive(voice_client)

    source = discord.FFmpegPCMAudio(str(audio_path))
    voice_client.play(source, after=cleanup)
    logger.info("Started voice playback in channel %s", voice_client.channel.id)


async def ensure_control_message_reaction(channel: discord.abc.Messageable) -> None:
    if not reaction_control_message_id or not hasattr(channel, "fetch_message"):
        return

    try:
        control_message = await channel.fetch_message(reaction_control_message_id)
    except discord.NotFound:
        logger.warning(
            "Configured reaction control message was not found in channel=%s message_id=%s",
            getattr(channel, "id", None),
            reaction_control_message_id,
        )
        return
    except discord.Forbidden:
        logger.exception("Missing permissions to fetch reaction control message")
        return
    except discord.HTTPException:
        logger.exception("Failed to fetch reaction control message")
        return

    existing_reaction = discord.utils.find(
        lambda reaction: str(reaction.emoji) == reaction_control_emoji,
        control_message.reactions,
    )
    if existing_reaction:
        return

    try:
        await control_message.add_reaction(reaction_control_emoji)
        logger.info(
            "Added control reaction emoji=%s to message_id=%s channel=%s",
            reaction_control_emoji,
            reaction_control_message_id,
            getattr(channel, "id", None),
        )
    except discord.Forbidden:
        logger.exception("Missing permissions to add control reaction")
    except discord.HTTPException:
        logger.exception("Failed to add control reaction")


async def remove_control_message_reaction(channel: discord.abc.Messageable) -> None:
    if not reaction_control_message_id or not hasattr(channel, "fetch_message") or not client.user:
        return

    try:
        control_message = await channel.fetch_message(reaction_control_message_id)
    except discord.NotFound:
        logger.warning(
            "Configured reaction control message was not found during cleanup channel=%s message_id=%s",
            getattr(channel, "id", None),
            reaction_control_message_id,
        )
        return
    except discord.Forbidden:
        logger.exception("Missing permissions to fetch reaction control message during cleanup")
        return
    except discord.HTTPException:
        logger.exception("Failed to fetch reaction control message during cleanup")
        return

    try:
        await control_message.remove_reaction(reaction_control_emoji, client.user)
        logger.info(
            "Removed control reaction emoji=%s from message_id=%s channel=%s",
            reaction_control_emoji,
            reaction_control_message_id,
            getattr(channel, "id", None),
        )
    except discord.Forbidden:
        logger.exception("Missing permissions to remove control reaction")
    except discord.HTTPException:
        logger.exception("Failed to remove control reaction")


async def run_speech_recognition_capture(
    voice_client: voice_recv.VoiceRecvClient,
    duration_seconds: int,
    guild_id: int,
    target_user: discord.abc.User,
) -> str:
    logger.info(
        "Starting speech recognition capture duration=%ss channel=%s target_user=%s",
        duration_seconds,
        voice_client.channel.id if voice_client.channel else None,
        target_user,
    )

    stop_idle_keepalive(voice_client)
    stop_voice_capture(voice_client)

    finished = asyncio.get_running_loop().create_future()
    stop_event = asyncio.Event()
    transcript_parts: list[str] = []
    recognition_errors: list[str] = []
    recording_path = create_recording_path(target_user)
    recording_params: wave._wave_params | None = None
    recording_pcm_chunks: list[bytes] = []
    tracked_ssrcs = {
        ssrc
        for ssrc, user_id in voice_client._ssrc_to_id.items()
        if user_id == target_user.id
    }
    logger.info(
        "Speech recognition capture targeting user_id=%s tracked_ssrcs=%s all_ssrc_map=%s",
        target_user.id,
        sorted(tracked_ssrcs),
        dict(voice_client._ssrc_to_id),
    )
    ACTIVE_RECORDING_SSRCS.update(tracked_ssrcs)
    for ssrc in tracked_ssrcs:
        CORRUPT_PACKET_COUNTS.pop(ssrc, None)

    def process_cb(recognizer, audio, user):
        try:
            if user is None or user.id != target_user.id:
                logger.info(
                    "Ignoring speech chunk for non-target user=%s target_user_id=%s",
                    user,
                    target_user.id,
                )
                return None
            audio_data = audio.get_wav_data()
            nonlocal recording_params
            recording_params, _ = append_wav_frames(audio_data, recording_pcm_chunks)
            boosted_audio_data, rms = amplify_wav_bytes(audio_data)
            model = get_whisper_model()
            segments, info = model.transcribe(
                io.BytesIO(boosted_audio_data),
                beam_size=1,
                vad_filter=False,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            logger.info(
                "Speech recognition process callback completed user=%s target_user_id=%s language=%s duration=%s chars=%s rms=%s",
                user,
                target_user.id,
                getattr(info, "language", None),
                getattr(info, "duration", None),
                len(text),
                rms,
            )
            return text or None
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            recognition_errors.append(message)
            logger.exception("Speech recognition process callback failed for user=%s", user)
            return None

    def text_cb(user, text):
        if user is None or user.id != target_user.id:
            logger.info(
                "Ignoring speech text for non-target user=%s target_user_id=%s text=%s",
                user,
                target_user.id,
                text,
            )
            return
        logger.info(
            "SpeechRecognitionSink text user=%s target_user_id=%s text=%s",
            user,
            target_user.id,
            text,
        )
        if text and text.strip():
            transcript_parts.append(text.strip())

    sink = SpeechRecognitionSink(
        process_cb=process_cb,
        text_cb=text_cb,
        default_recognizer="whisper",
        phrase_time_limit=duration_seconds,
        ignore_silence_packets=True,
    )
    ACTIVE_RECORDINGS[guild_id] = {
        "voice_client": voice_client,
        "stop_event": stop_event,
        "tracked_ssrcs": tracked_ssrcs,
        "mode": "speechrecognition",
        "target_user_id": target_user.id,
    }

    def after_recording(exc: Exception | None) -> None:
        try:
            sink.cleanup()
        except Exception:
            logger.exception("Failed cleaning up speech recognition sink")

        if exc:
            logger.exception("Speech recognition callback reported an error", exc_info=exc)
        if not finished.done():
            finished.set_result(exc)

    try:
        voice_client.listen(sink, after=after_recording)
    except Exception:
        logger.exception("Failed to start speech recognition capture")
        try:
            sink.cleanup()
        except Exception:
            logger.exception("Failed cleaning up speech recognition sink after startup failure")
        raise

    try:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=duration_seconds)
            logger.info("Speech recognition capture stopped early for guild=%s", guild_id)
        except asyncio.TimeoutError:
            logger.info("Speech recognition capture completed full duration for guild=%s", guild_id)
    finally:
        stop_voice_capture(voice_client)
        callback_error = await finished
        if voice_client.is_connected():
            start_idle_keepalive(voice_client)
        session = ACTIVE_RECORDINGS.pop(guild_id, None)
        tracked_ssrcs = session.get("tracked_ssrcs", set()) if session else set()
        total_corrupt_packets = 0
        for ssrc in tracked_ssrcs:
            ACTIVE_RECORDING_SSRCS.discard(ssrc)
            total_corrupt_packets += CORRUPT_PACKET_COUNTS.pop(ssrc, 0)
        if total_corrupt_packets:
            logger.warning(
                "Speech recognition capture encountered %s corrupt Opus packets across %s SSRC(s) for target_user_id=%s",
                total_corrupt_packets,
                len(tracked_ssrcs),
                session.get("target_user_id") if session else None,
            )
        if callback_error and not is_recoverable_capture_error(callback_error):
            raise RuntimeError(f"Speech recognition capture ended with an error: {callback_error}")
        if recognition_errors and not transcript_parts:
            raise RuntimeError(f"Speech recognizer failed: {recognition_errors[-1]}")

    if save_recording_wav(recording_path, recording_params, recording_pcm_chunks):
        logger.info(
            "Saved recording to %s size_bytes=%s chunks=%s target_user_id=%s",
            recording_path,
            recording_path.stat().st_size,
            len(recording_pcm_chunks),
            target_user.id,
        )
    else:
        logger.warning("No recording audio was saved for target_user_id=%s", target_user.id)

    transcript = " ".join(transcript_parts).strip()
    logger.info("Speech recognition capture produced chars=%s", len(transcript))
    return transcript


load_dotenv()
logger = setup_logging()
log_voice_backend_status()
log_phase3_backend_status()
patch_voice_recv_decoder()

token = get_required_env("DISCORD_BOT_TOKEN", "DISCORD_ECHO_TOKEN")
text_channel_id = get_env_value("BOT_TEXT_CHANNEL_ID", "BOT_ALLOWED_CHANNEL_ID", "ECHO_CHAMBER_CHANNEL_ID")
if text_channel_id:
    text_channel_id = int(text_channel_id)
reaction_control_message_id = get_env_value("BOT_REACTION_MESSAGE_ID")
if reaction_control_message_id:
    reaction_control_message_id = int(reaction_control_message_id)
reaction_control_emoji = get_env_value("BOT_REACTION_EMOJI", default="🎙️")
reaction_record_seconds = int(
    get_env_value("BOT_REACTION_RECORD_SECONDS", default=str(DEFAULT_REACTION_RECORD_SECONDS))
)
conversation_logging_enabled = env_flag("ENABLE_CONVERSATION_LOGGING")

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True
intents.reactions = True

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

    if text_channel_id is not None and message.channel.id != text_channel_id:
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
        voice_client = get_current_voice_client(message.guild)
        if message.guild and voice_client:
            logger.info(
                "Manual voice disconnect requested guild=%s state=%s",
                message.guild.id,
                voice_state_snapshot(voice_client),
            )
            stop_idle_keepalive(voice_client)
            await voice_client.disconnect(force=True)
            await remove_control_message_reaction(message.channel)
            logger.info("Disconnected from voice in guild %s", message.guild.id)
            await message.channel.send("Left the voice channel.")
        else:
            await message.channel.send("I'm not in a voice channel right now.")
        return

    if command == "!clear":
        await handle_clear_command(message, content)
        return

    if command in {"!record", "!talk"}:
        duration_seconds = DEFAULT_RECORD_SECONDS
        if len(parts) == 2:
            try:
                duration_seconds = int(parts[1])
            except ValueError:
                await send_error(
                    message.channel,
                    f"Usage: `{command}` or `{command} {DEFAULT_RECORD_SECONDS}`",
                )
                return

        if duration_seconds < 1 or duration_seconds > MAX_RECORD_SECONDS:
            await send_error(
                message.channel,
                f"`{command}` accepts a number between 1 and {MAX_RECORD_SECONDS} seconds.",
            )
            return

        await handle_record_or_talk(
            message,
            duration_seconds,
            mode="talk" if command == "!talk" else "record",
        )
        return

    if command == "!stop":
        stopped_anything = False

        if message.guild and stop_active_recording(message.guild.id):
            stopped_anything = True

        voice_client = get_current_voice_client(message.guild)
        if voice_client and voice_client.is_playing():
            logger.info(
                "Stop requested for active playback in guild=%s state=%s",
                message.guild.id,
                voice_state_snapshot(voice_client),
            )
            stop_idle_keepalive(voice_client)
            if voice_client.is_playing():
                voice_client.stop()
            if voice_client.is_connected():
                start_idle_keepalive(voice_client)
            stopped_anything = True

        if not stopped_anything:
            await send_error(message.channel, "There is no active recording or playback to stop.")
        return

    if command != "!ask":
        return

    question = content[4:].strip()
    if not question:
        await message.channel.send("Usage: `!ask your question here`")
        return

    await respond_with_ollama(
        message,
        question,
        speak_reply=True,
        log_source="text",
    )


def is_matching_control_reaction(payload: discord.RawReactionActionEvent) -> bool:
    return bool(
        reaction_control_message_id
        and payload.message_id == reaction_control_message_id
        and str(payload.emoji) == reaction_control_emoji
    )


async def fetch_reaction_context(payload: discord.RawReactionActionEvent):
    if not payload.guild_id:
        return None

    guild = client.get_guild(payload.guild_id)
    if guild is None:
        logger.warning("Could not resolve guild for reaction payload guild_id=%s", payload.guild_id)
        return None

    channel = client.get_channel(payload.channel_id)
    if channel is None:
        try:
            channel = await client.fetch_channel(payload.channel_id)
        except Exception:
            logger.exception("Could not fetch channel for reaction payload channel_id=%s", payload.channel_id)
            return None

    member = payload.member
    if member is None:
        member = guild.get_member(payload.user_id)
    if member is None:
        try:
            member = await guild.fetch_member(payload.user_id)
        except Exception:
            logger.exception("Could not fetch member for reaction payload user_id=%s", payload.user_id)
            return None

    return SimpleNamespace(guild=guild, channel=channel, author=member)


async def run_reaction_voice_loop(context, duration_seconds: int) -> None:
    try:
        await handle_record_or_talk(context, duration_seconds, mode="talk")
    finally:
        if context.guild:
            REACTION_RECORDING_TASKS.pop(context.guild.id, None)


@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
    if client.user and payload.user_id == client.user.id:
        return
    if not is_matching_control_reaction(payload):
        return

    context = await fetch_reaction_context(payload)
    if context is None:
        return

    if context.guild.id in REACTION_RECORDING_TASKS:
        logger.info(
            "Ignoring reaction start because a reaction-controlled recording task is already active guild=%s user=%s",
            context.guild.id,
            payload.user_id,
        )
        return

    if context.guild.id in ACTIVE_RECORDINGS:
        logger.info(
            "Ignoring reaction start because a recording is already active guild=%s user=%s",
            context.guild.id,
            payload.user_id,
        )
        return

    logger.info(
        "Starting reaction-controlled voice loop guild=%s user=%s channel=%s emoji=%s duration=%s",
        context.guild.id,
        payload.user_id,
        context.channel.id,
        payload.emoji,
        reaction_record_seconds,
    )
    REACTION_RECORDING_TASKS[context.guild.id] = asyncio.create_task(
        run_reaction_voice_loop(context, reaction_record_seconds)
    )


@client.event
async def on_raw_reaction_remove(payload: discord.RawReactionActionEvent) -> None:
    if client.user and payload.user_id == client.user.id:
        return
    if not is_matching_control_reaction(payload):
        return
    if not payload.guild_id:
        return

    session = ACTIVE_RECORDINGS.get(payload.guild_id)
    if not session:
        logger.info(
            "Ignoring reaction stop because no recording is active guild=%s user=%s",
            payload.guild_id,
            payload.user_id,
        )
        return

    if session.get("target_user_id") != payload.user_id:
        logger.info(
            "Ignoring reaction stop from non-owner guild=%s owner=%s remover=%s",
            payload.guild_id,
            session.get("target_user_id"),
            payload.user_id,
        )
        return

    logger.info(
        "Stopping reaction-controlled recording guild=%s user=%s emoji=%s",
        payload.guild_id,
        payload.user_id,
        payload.emoji,
    )
    stop_active_recording(payload.guild_id)


if __name__ == "__main__":
    client.run(token, log_handler=None)
