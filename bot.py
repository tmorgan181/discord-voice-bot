import asyncio
import array
import io
import json
import logging
import math
import os
import re
import uuid
import wave
from collections import defaultdict
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
import speech_recognition as sr

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
DEFAULT_REACTION_STOP_DELAY_MS = 750
DEFAULT_AUTO_LISTEN_SILENCE_SECONDS = 1.5
DEFAULT_AUTO_LISTEN_PHRASE_LIMIT_SECONDS = 30
DEFAULT_AUTO_LISTEN_MIN_DISTINCT_WORDS = 3
DEFAULT_AUTO_LISTEN_MIN_AUDIO_SECONDS = 0.9
DEFAULT_AUTO_LISTEN_WINDOW_MS = 20
DEFAULT_AUTO_LISTEN_VOICED_WINDOW_RMS = 900.0
DEFAULT_AUTO_LISTEN_MIN_VOICED_WINDOWS = 8
DEFAULT_AUTO_LISTEN_MIN_VOICED_RATIO = 0.2
DEFAULT_AUTO_LISTEN_MAX_ZERO_CROSSING_RATE = 0.3
DEFAULT_AUDIO_GAIN = 1.15
DEFAULT_TARGET_RMS = 14000.0
DEFAULT_MAX_AUDIO_GAIN = 6.0
DEFAULT_WHISPER_MODEL = "base.en"
DEFAULT_WHISPER_BEAM_SIZE = 1
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
ACTIVE_CONVERSATIONS: dict[int, dict] = {}
AUTO_LISTEN_SESSIONS: dict[int, dict] = {}
AUTO_LISTEN_ENABLED_GUILDS: set[int] = set()


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


def get_history_max_turns() -> int:
    raw_value = get_env_value("BOT_HISTORY_MAX_TURNS", default="12")
    try:
        return max(1, int(raw_value or "12"))
    except ValueError:
        return 12


def get_env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def get_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def get_auto_listen_silence_seconds() -> float:
    return max(0.5, get_env_float("BOT_AUTO_LISTEN_SILENCE_SECONDS", DEFAULT_AUTO_LISTEN_SILENCE_SECONDS))


def get_auto_listen_phrase_limit_seconds() -> int:
    return max(3, get_env_int("BOT_AUTO_LISTEN_PHRASE_LIMIT_SECONDS", DEFAULT_AUTO_LISTEN_PHRASE_LIMIT_SECONDS))


def get_auto_listen_min_distinct_words() -> int:
    return max(
        1,
        get_env_int(
            "BOT_AUTO_LISTEN_MIN_DISTINCT_WORDS",
            DEFAULT_AUTO_LISTEN_MIN_DISTINCT_WORDS,
        ),
    )


def get_auto_listen_min_audio_seconds() -> float:
    return max(0.1, get_env_float("BOT_AUTO_LISTEN_MIN_AUDIO_SECONDS", DEFAULT_AUTO_LISTEN_MIN_AUDIO_SECONDS))


def get_auto_listen_voiced_window_rms() -> float:
    return max(1.0, get_env_float("BOT_AUTO_LISTEN_VOICED_WINDOW_RMS", DEFAULT_AUTO_LISTEN_VOICED_WINDOW_RMS))


def get_auto_listen_min_voiced_windows() -> int:
    return max(1, get_env_int("BOT_AUTO_LISTEN_MIN_VOICED_WINDOWS", DEFAULT_AUTO_LISTEN_MIN_VOICED_WINDOWS))


def get_auto_listen_min_voiced_ratio() -> float:
    return max(0.0, min(1.0, get_env_float("BOT_AUTO_LISTEN_MIN_VOICED_RATIO", DEFAULT_AUTO_LISTEN_MIN_VOICED_RATIO)))


def get_auto_listen_max_zero_crossing_rate() -> float:
    return max(
        0.0,
        min(1.0, get_env_float("BOT_AUTO_LISTEN_MAX_ZERO_CROSSING_RATE", DEFAULT_AUTO_LISTEN_MAX_ZERO_CROSSING_RATE)),
    )


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


def reset_conversation_history(guild_id: int, channel_id: int | None) -> None:
    ACTIVE_CONVERSATIONS[guild_id] = {
        "channel_id": channel_id,
        "turns": [],
    }
    logger.info(
        "Reset conversation history for guild=%s channel=%s",
        guild_id,
        channel_id,
    )


def clear_conversation_history(guild_id: int) -> None:
    if ACTIVE_CONVERSATIONS.pop(guild_id, None) is not None:
        logger.info("Cleared conversation history for guild=%s", guild_id)


def get_active_conversation_channel_id(guild: discord.Guild | None) -> int | None:
    voice_client = get_current_voice_client(guild)
    if voice_client and voice_client.channel:
        return voice_client.channel.id
    return None


def get_conversation_history_text(message: discord.Message) -> str:
    if not message.guild:
        return ""

    session = ACTIVE_CONVERSATIONS.get(message.guild.id)
    if not session:
        return ""
    if session.get("channel_id") != get_active_conversation_channel_id(message.guild):
        return ""

    turns = session.get("turns", [])
    if not turns:
        return ""

    history_lines = ["Conversation so far in this voice session:"]
    for turn in turns[-get_history_max_turns():]:
        history_lines.append(f"{turn['role']}: {turn['content']}")
    return "\n".join(history_lines)


def add_conversation_turn(guild_id: int | None, channel_id: int | None, role: str, content: str) -> None:
    if guild_id is None or channel_id is None or not content.strip():
        return

    session = ACTIVE_CONVERSATIONS.get(guild_id)
    if session is None or session.get("channel_id") != channel_id:
        reset_conversation_history(guild_id, channel_id)
        session = ACTIVE_CONVERSATIONS[guild_id]

    session["turns"].append(
        {
            "role": role,
            "content": content.strip(),
        }
    )
    max_turns = get_history_max_turns() * 2
    if len(session["turns"]) > max_turns:
        session["turns"] = session["turns"][-max_turns:]


def ask_ollama_with_context(message: discord.Message, prompt: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    system_prompt = build_system_prompt(message)

    full_prompt = prompt
    history_text = get_conversation_history_text(message)
    if history_text:
        full_prompt = f"{history_text}\n\nCurrent user message:\n{prompt}"

    payload = {
        "model": model,
        "prompt": full_prompt,
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


def get_ollama_display_target() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL).rstrip("/")


class TunedSpeechRecognitionSink(SpeechRecognitionSink):
    def __init__(
        self,
        *,
        process_cb=None,
        text_cb=None,
        default_recognizer: str = "whisper",
        phrase_time_limit: int = 10,
        ignore_silence_packets: bool = True,
        pause_threshold: float = DEFAULT_AUTO_LISTEN_SILENCE_SECONDS,
    ) -> None:
        super().__init__(
            process_cb=process_cb,
            text_cb=text_cb,
            default_recognizer=default_recognizer,
            phrase_time_limit=phrase_time_limit,
            ignore_silence_packets=ignore_silence_packets,
        )
        self._pause_threshold = pause_threshold
        self._stream_data = defaultdict(self._make_stream_data)

    def _make_stream_data(self):
        recognizer = sr.Recognizer()
        recognizer.pause_threshold = self._pause_threshold
        recognizer.non_speaking_duration = min(self._pause_threshold, 0.5)
        recognizer.dynamic_energy_threshold = True
        return {
            "stopper": None,
            "recognizer": recognizer,
            "buffer": array.array("B"),
        }


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
    add_conversation_turn(
        message.guild.id if message.guild else None,
        get_active_conversation_channel_id(message.guild),
        "user",
        prompt,
    )

    async with message.channel.typing():
        try:
            answer = await asyncio.to_thread(ask_ollama_with_context, message, prompt)
            logger.info("Ollama answered successfully for user %s source=%s", message.author.id, log_source)
        except requests.ConnectionError as exc:
            ollama_target = get_ollama_display_target()
            logger.warning(
                "Ollama connection failed target=%s user=%s source=%s error=%s",
                ollama_target,
                message.author.id,
                log_source,
                exc,
            )
            await send_error(
                message.channel,
                f"I couldn't reach Ollama at `{ollama_target}`. Make sure `ollama serve` is running, then try again.",
            )
            return None
        except requests.Timeout as exc:
            ollama_target = get_ollama_display_target()
            logger.warning(
                "Ollama request timed out target=%s user=%s source=%s error=%s",
                ollama_target,
                message.author.id,
                log_source,
                exc,
            )
            await send_error(
                message.channel,
                f"Ollama at `{ollama_target}` took too long to respond. Try again in a moment.",
            )
            return None
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            ollama_target = get_ollama_display_target()
            logger.warning(
                "Ollama returned HTTP error target=%s status=%s user=%s source=%s",
                ollama_target,
                status_code,
                message.author.id,
                log_source,
            )
            await send_error(
                message.channel,
                f"Ollama returned an HTTP error (`{status_code}`) from `{ollama_target}`.",
            )
            return None
        except requests.RequestException as exc:
            ollama_target = get_ollama_display_target()
            logger.warning(
                "Ollama request failed target=%s user=%s source=%s error=%s",
                ollama_target,
                message.author.id,
                log_source,
                exc,
            )
            await send_error(
                message.channel,
                f"Ollama request failed at `{ollama_target}`. Check the bot log for details.",
            )
            return None
        except Exception:
            logger.exception("Unexpected Ollama error")
            await send_error(
                message.channel,
                "Something went wrong while asking Ollama.",
            )
            return None

    add_conversation_turn(
        message.guild.id if message.guild else None,
        get_active_conversation_channel_id(message.guild),
        "assistant",
        answer,
    )
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
    status_message: str | None = None,
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
        status_text = status_message
        if status_text is None:
            base_status = "Recording and listening" if mode == "talk" else "Recording"
            status_text = f"{base_status} for {duration_seconds} seconds..."
        await context.channel.send(status_text)
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
        await ensure_auto_listen(context.guild, context.channel)
        return

    if transcript == "[No speech detected]":
        await context.channel.send(f"Transcript: {transcript}")
        await ensure_auto_listen(context.guild, context.channel)
        return

    await context.channel.send(f"Heard: {transcript}")
    await respond_with_ollama(
        context,
        transcript,
        send_prefix=None,
        speak_reply=True,
        log_source="voice",
    )
    await ensure_auto_listen(context.guild, context.channel)


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


def compute_pcm_peak_abs(frames: bytes, sample_width: int) -> int:
    if not frames:
        return 0
    if sample_width != 2:
        return 0

    samples = array.array("h")
    samples.frombytes(frames)
    if not samples:
        return 0

    return max(abs(sample) for sample in samples)


def compute_zero_crossing_rate(frames: bytes, sample_width: int) -> float:
    if not frames or sample_width != 2:
        return 0.0

    samples = array.array("h")
    samples.frombytes(frames)
    if len(samples) < 2:
        return 0.0

    crossings = 0
    previous_sign = 1 if samples[0] >= 0 else -1
    for sample in samples[1:]:
        sign = 1 if sample >= 0 else -1
        if sign != previous_sign:
            crossings += 1
        previous_sign = sign

    return crossings / max(1, len(samples) - 1)


def compute_audio_duration_seconds(frame_count: int, sample_rate: int) -> float:
    if frame_count <= 0 or sample_rate <= 0:
        return 0.0
    return frame_count / sample_rate


def count_voiced_windows(
    frames: bytes,
    sample_width: int,
    sample_rate: int,
    *,
    window_ms: int = DEFAULT_AUTO_LISTEN_WINDOW_MS,
    rms_threshold: float = DEFAULT_AUTO_LISTEN_VOICED_WINDOW_RMS,
) -> int:
    if not frames or sample_width != 2 or sample_rate <= 0:
        return 0

    window_frames = max(1, int(sample_rate * (window_ms / 1000)))
    window_bytes = window_frames * sample_width
    voiced = 0

    for offset in range(0, len(frames), window_bytes):
        chunk = frames[offset : offset + window_bytes]
        if compute_pcm_rms(chunk, sample_width) >= rms_threshold:
            voiced += 1

    return voiced


def analyze_voiced_windows(
    frames: bytes,
    sample_width: int,
    sample_rate: int,
    *,
    window_ms: int = DEFAULT_AUTO_LISTEN_WINDOW_MS,
    rms_threshold: float = DEFAULT_AUTO_LISTEN_VOICED_WINDOW_RMS,
) -> tuple[int, int, float]:
    if not frames or sample_width != 2 or sample_rate <= 0:
        return 0, 0, 0.0

    window_frames = max(1, int(sample_rate * (window_ms / 1000)))
    window_bytes = window_frames * sample_width
    voiced = 0
    total = 0
    voiced_zero_crossing_rates: list[float] = []

    for offset in range(0, len(frames), window_bytes):
        chunk = frames[offset : offset + window_bytes]
        if not chunk:
            continue
        total += 1
        if compute_pcm_rms(chunk, sample_width) >= rms_threshold:
            voiced += 1
            voiced_zero_crossing_rates.append(compute_zero_crossing_rate(chunk, sample_width))

    average_voiced_zcr = (
        sum(voiced_zero_crossing_rates) / len(voiced_zero_crossing_rates)
        if voiced_zero_crossing_rates
        else 0.0
    )
    return voiced, total, average_voiced_zcr


def should_accept_auto_audio_chunk(audio_bytes: bytes) -> tuple[bool, str, dict]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())

    duration_seconds = compute_audio_duration_seconds(params.nframes, params.framerate)
    rms = compute_pcm_rms(frames, params.sampwidth)
    voiced_windows, total_windows, average_voiced_zcr = analyze_voiced_windows(
        frames,
        params.sampwidth,
        params.framerate,
        window_ms=DEFAULT_AUTO_LISTEN_WINDOW_MS,
        rms_threshold=get_auto_listen_voiced_window_rms(),
    )
    voiced_ratio = (voiced_windows / total_windows) if total_windows else 0.0

    stats = {
        "duration_seconds": round(duration_seconds, 3),
        "rms": round(rms, 1),
        "voiced_windows": voiced_windows,
        "voiced_ratio": round(voiced_ratio, 3),
        "avg_voiced_zcr": round(average_voiced_zcr, 3),
    }

    if duration_seconds < get_auto_listen_min_audio_seconds():
        return False, "too_short", stats
    if voiced_windows < get_auto_listen_min_voiced_windows():
        return False, "not_enough_voiced_windows", stats
    if voiced_ratio < get_auto_listen_min_voiced_ratio():
        return False, "not_enough_voiced_ratio", stats
    if average_voiced_zcr > get_auto_listen_max_zero_crossing_rate():
        return False, "too_noisy", stats
    return True, "accepted", stats


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


def merge_transcript_text(existing_text: str, new_text: str) -> str:
    existing = existing_text.strip()
    incoming = new_text.strip()
    if not incoming:
        return existing
    if not existing:
        return incoming

    existing_lower = existing.lower()
    incoming_lower = incoming.lower()

    if incoming_lower == existing_lower:
        return existing
    if incoming_lower in existing_lower:
        return existing
    if existing_lower.endswith(incoming_lower):
        return existing

    max_overlap = min(len(existing), len(incoming))
    for overlap in range(max_overlap, 0, -1):
        if existing_lower[-overlap:] == incoming_lower[:overlap]:
            return f"{existing}{incoming[overlap:]}".strip()

    return f"{existing} {incoming}".strip()


def extract_distinct_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {word for word in words if len(word) >= 2}


def amplify_wav_bytes(audio_bytes: bytes) -> tuple[bytes, float, float]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())

    rms = compute_pcm_rms(frames, params.sampwidth)
    peak = compute_pcm_peak_abs(frames, params.sampwidth)
    manual_gain = get_env_float("WHISPER_AUDIO_GAIN", DEFAULT_AUDIO_GAIN)
    target_rms = get_env_float("WHISPER_TARGET_RMS", DEFAULT_TARGET_RMS)
    max_gain = max(1.0, get_env_float("WHISPER_MAX_GAIN", DEFAULT_MAX_AUDIO_GAIN))

    adaptive_gain = 1.0
    if rms > 0 and target_rms > 0:
        adaptive_gain = max(1.0, target_rms / rms)

    headroom_gain = max_gain
    if peak > 0:
        headroom_gain = 32767 / peak

    applied_gain = min(max_gain, adaptive_gain * manual_gain, headroom_gain)
    boosted_frames = scale_pcm_frames(frames, params.sampwidth, applied_gain)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setparams(params)
        writer.writeframes(boosted_frames)
    return buffer.getvalue(), float(rms), float(applied_gain)


def get_whisper_model() -> WhisperModel:
    global WHISPER_MODEL_INSTANCE
    if WHISPER_MODEL_INSTANCE is None:
        model_name = os.getenv("WHISPER_MODEL", DEFAULT_WHISPER_MODEL)
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


def transcribe_wav_bytes(audio_data: bytes) -> tuple[str, object, float, float]:
    boosted_audio_data, rms, applied_gain = amplify_wav_bytes(audio_data)
    model = get_whisper_model()
    segments, info = model.transcribe(
        io.BytesIO(boosted_audio_data),
        beam_size=max(1, get_env_int("WHISPER_BEAM_SIZE", DEFAULT_WHISPER_BEAM_SIZE)),
        vad_filter=False,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return text, info, rms, applied_gain


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
        clear_conversation_history(message.guild.id)
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


def get_bot_voice_channel(guild: discord.Guild | None) -> discord.abc.GuildChannel | None:
    if guild is None or client.user is None:
        return None
    member = guild.get_member(client.user.id)
    if member is None or member.voice is None:
        return None
    return member.voice.channel


async def fetch_bot_voice_channel(guild: discord.Guild | None) -> discord.abc.GuildChannel | None:
    if guild is None or client.user is None:
        return None

    member = guild.get_member(client.user.id)
    if member is None:
        try:
            member = await guild.fetch_member(client.user.id)
        except Exception:
            logger.exception("Failed to fetch bot member state for guild %s", guild.id)
            return None
    else:
        try:
            member = await guild.fetch_member(client.user.id)
        except Exception:
            pass

    if member.voice is None:
        return None
    return member.voice.channel


async def fetch_bot_member(guild: discord.Guild | None) -> discord.Member | None:
    if guild is None or client.user is None:
        return None

    member = guild.get_member(client.user.id)
    if member is None:
        try:
            member = await guild.fetch_member(client.user.id)
        except Exception:
            logger.exception("Failed to fetch bot member for guild %s", guild.id)
            return None
    else:
        try:
            member = await guild.fetch_member(client.user.id)
        except Exception:
            pass
    return member


async def force_leave_guild_voice(guild: discord.Guild, *, reason: str) -> bool:
    voice_client = get_current_voice_client(guild)
    bot_voice_channel = await fetch_bot_voice_channel(guild)
    bot_member = await fetch_bot_member(guild)
    if voice_client is None and bot_voice_channel is None:
        return False

    logger.info(
        "Forcing bot voice disconnect guild=%s reason=%s local_client=%s bot_voice_channel=%s",
        guild.id,
        reason,
        voice_state_snapshot(voice_client),
        bot_voice_channel.id if bot_voice_channel else None,
    )

    stop_auto_listen(guild.id, disable=True)

    if voice_client is not None:
        try:
            stop_idle_keepalive(voice_client)
            await voice_client.disconnect(force=True)
        except Exception:
            logger.exception("Failed to disconnect active voice client for guild %s", guild.id)

    if await fetch_bot_voice_channel(guild) is not None:
        try:
            await guild.change_voice_state(channel=None, self_mute=False, self_deaf=False)
            logger.info("Issued guild.change_voice_state disconnect for guild=%s reason=%s", guild.id, reason)
        except Exception:
            logger.exception("Failed to clear lingering guild voice state for guild %s", guild.id)

    if await fetch_bot_voice_channel(guild) is not None and bot_member is not None:
        try:
            await bot_member.move_to(None, reason=f"codex:{reason}")
            logger.info("Issued member.move_to(None) disconnect for guild=%s reason=%s", guild.id, reason)
        except Exception:
            logger.exception("Failed to move bot member out of voice for guild %s", guild.id)

    remaining_voice_channel = None
    for _ in range(6):
        remaining_voice_channel = await fetch_bot_voice_channel(guild)
        if remaining_voice_channel is None:
            break
        await asyncio.sleep(0.5)

    if remaining_voice_channel is not None:
        logger.warning(
            "Bot still appears connected after disconnect attempt guild=%s channel=%s reason=%s",
            guild.id,
            remaining_voice_channel.id,
            reason,
        )

    clear_conversation_history(guild.id)
    await remove_control_message_reaction_for_guild(guild)
    return remaining_voice_channel is None


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
                        if guild.id not in ACTIVE_CONVERSATIONS:
                            reset_conversation_history(guild.id, target_channel.id)
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
                    if guild.id not in ACTIVE_CONVERSATIONS:
                        reset_conversation_history(guild.id, target_channel.id)
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
                    reset_conversation_history(guild.id, target_channel.id)
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
                reset_conversation_history(guild.id, target_channel.id)
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
    guild = getattr(voice_client, "guild", None)
    guild_id = guild.id if guild else None
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
            if guild is not None and guild.id in AUTO_LISTEN_ENABLED_GUILDS:
                try:
                    asyncio.run_coroutine_threadsafe(
                        ensure_auto_listen(guild),
                        client.loop,
                    )
                except Exception:
                    logger.exception("Failed to resume auto-listen after TTS playback")

    if guild_id is not None:
        stop_auto_listen(guild_id)
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


async def remove_control_message_reaction_for_guild(guild: discord.Guild) -> None:
    channel = await resolve_text_channel_for_guild(guild)
    if channel is None:
        logger.warning("Could not resolve a text channel for control reaction cleanup guild=%s", guild.id)
        return
    await remove_control_message_reaction(channel)


async def resolve_text_channel_for_guild(
    guild: discord.Guild, fallback_channel: discord.abc.Messageable | None = None
):
    if text_channel_id:
        channel = client.get_channel(text_channel_id)
        if channel is None:
            try:
                channel = await client.fetch_channel(text_channel_id)
            except Exception:
                logger.exception("Could not fetch configured text channel text_channel_id=%s", text_channel_id)
                channel = None
        if channel is not None:
            return channel
    return fallback_channel


def stop_auto_listen(guild_id: int, *, disable: bool = False) -> bool:
    if disable:
        AUTO_LISTEN_ENABLED_GUILDS.discard(guild_id)
    session = AUTO_LISTEN_SESSIONS.pop(guild_id, None)
    if not session:
        return False

    voice_client = session.get("voice_client")
    if isinstance(voice_client, voice_recv.VoiceRecvClient) and voice_client.is_listening():
        logger.info("Stopping auto-listen for guild=%s", guild_id)
        voice_client.stop_listening()
    return True


async def handle_auto_transcript(guild_id: int, user, text: str) -> None:
    session = AUTO_LISTEN_SESSIONS.get(guild_id)
    if not session or not text.strip():
        return

    if client.user and user.id == client.user.id:
        return

    last_text = session.get("last_text", "")
    if text.strip().lower() == str(last_text).strip().lower():
        logger.info("Ignoring duplicate auto transcript guild=%s user=%s text=%s", guild_id, user.id, text)
        return

    distinct_words = extract_distinct_words(text)
    min_distinct_words = get_auto_listen_min_distinct_words()
    if len(distinct_words) < min_distinct_words:
        logger.info(
            "Ignoring short auto transcript guild=%s user=%s distinct_words=%s min_required=%s text=%s",
            guild_id,
            user.id,
            len(distinct_words),
            min_distinct_words,
            text,
        )
        return

    if session.get("busy"):
        logger.info("Ignoring auto transcript while another auto response is in flight guild=%s user=%s", guild_id, user.id)
        return

    guild = session["guild"]
    fallback_channel = session.get("text_channel")
    channel = await resolve_text_channel_for_guild(guild, fallback_channel)
    if channel is None:
        logger.warning("Auto-listen could not resolve a text channel for guild=%s", guild_id)
        return

    session["busy"] = True
    session["last_text"] = text.strip()
    stop_auto_listen(guild_id)
    try:
        await channel.send(f"Heard {getattr(user, 'display_name', user)}: {text.strip()}")
        message_like = SimpleNamespace(guild=guild, channel=channel, author=user)
        await respond_with_ollama(
            message_like,
            text.strip(),
            speak_reply=True,
            log_source="auto",
        )
    finally:
        current = AUTO_LISTEN_SESSIONS.get(guild_id)
        if current is not None:
            current["busy"] = False


async def ensure_auto_listen(guild: discord.Guild | None, fallback_channel=None) -> None:
    if guild is None:
        return
    if guild.id not in AUTO_LISTEN_ENABLED_GUILDS:
        return
    if guild.id in ACTIVE_RECORDINGS:
        return

    voice_client = get_current_voice_client(guild)
    if not isinstance(voice_client, voice_recv.VoiceRecvClient) or not voice_client.is_connected():
        return

    existing = AUTO_LISTEN_SESSIONS.get(guild.id)
    if existing and existing.get("voice_client") is voice_client and voice_client.is_listening():
        if fallback_channel is not None:
            existing["text_channel"] = fallback_channel
        return

    stop_auto_listen(guild.id)
    text_channel = await resolve_text_channel_for_guild(guild, fallback_channel)
    pause_threshold = get_auto_listen_silence_seconds()
    phrase_limit = get_auto_listen_phrase_limit_seconds()

    def process_cb(recognizer, audio, user):
        try:
            if user is None or (client.user and user.id == client.user.id):
                return None
            audio_data = audio.get_wav_data()
            accepted, reason, stats = should_accept_auto_audio_chunk(audio_data)
            if not accepted:
                logger.info(
                    "Rejected auto-listen audio chunk guild=%s user=%s reason=%s stats=%s",
                    guild.id,
                    user.id,
                    reason,
                    stats,
                )
                return None
            text, info, rms, applied_gain = transcribe_wav_bytes(audio_data)
            logger.info(
                "Auto-listen speech callback completed guild=%s user=%s language=%s duration=%s chars=%s rms=%s gain=%s",
                guild.id,
                user.id,
                getattr(info, "language", None),
                getattr(info, "duration", None),
                len(text),
                rms,
                round(applied_gain, 3),
            )
            return text or None
        except Exception:
            logger.exception("Auto-listen speech callback failed guild=%s user=%s", guild.id, getattr(user, "id", None))
            return None

    def text_cb(user, text):
        if user is None or not text or not text.strip():
            return
        logger.info("Auto-listen text guild=%s user=%s text=%s", guild.id, user.id, text)
        asyncio.run_coroutine_threadsafe(handle_auto_transcript(guild.id, user, text), client.loop)

    sink = TunedSpeechRecognitionSink(
        process_cb=process_cb,
        text_cb=text_cb,
        default_recognizer="whisper",
        phrase_time_limit=phrase_limit,
        ignore_silence_packets=True,
        pause_threshold=pause_threshold,
    )

    AUTO_LISTEN_SESSIONS[guild.id] = {
        "voice_client": voice_client,
        "guild": guild,
        "text_channel": text_channel,
        "busy": False,
        "last_text": "",
    }

    def after_auto_listen(exc: Exception | None) -> None:
        try:
            sink.cleanup()
        except Exception:
            logger.exception("Failed cleaning up auto-listen sink guild=%s", guild.id)
        if exc:
            logger.exception("Auto-listen callback reported an error guild=%s", guild.id, exc_info=exc)
        current = AUTO_LISTEN_SESSIONS.get(guild.id)
        if current and current.get("voice_client") is voice_client and guild.id not in ACTIVE_RECORDINGS:
            AUTO_LISTEN_SESSIONS.pop(guild.id, None)

    voice_client.listen(sink, after=after_auto_listen)
    logger.info(
        "Started auto-listen guild=%s voice_channel=%s pause_threshold=%s phrase_limit=%s text_channel=%s",
        guild.id,
        voice_client.channel.id if voice_client.channel else None,
        pause_threshold,
        phrase_limit,
        getattr(text_channel, "id", None),
    )


async def enable_hands_free_mode(context) -> bool:
    if not context.guild:
        await send_error(context.channel, "Hands-free mode only works inside a server.")
        return False

    AUTO_LISTEN_ENABLED_GUILDS.add(context.guild.id)

    try:
        voice_client = await ensure_voice_client(context)
    except discord.ClientException:
        logger.exception("Discord voice join failed while enabling hands-free mode")
        await send_error(
            context.channel,
            "I couldn't join the voice channel after several retries. Check my Connect/Speak permissions and try again.",
        )
        return False

    if voice_client is None:
        return False

    await ensure_auto_listen(context.guild, context.channel)
    await context.channel.send(
        f"Hands-free mode is live in `{voice_client.channel.name}`. I'll listen, respond, talk, then listen again."
    )
    return True


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

    stop_auto_listen(guild_id)
    stop_idle_keepalive(voice_client)
    stop_voice_capture(voice_client)

    finished = asyncio.get_running_loop().create_future()
    stop_event = asyncio.Event()
    transcript_text = ""
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
            text, info, rms, applied_gain = transcribe_wav_bytes(audio_data)
            logger.info(
                "Speech recognition process callback completed user=%s target_user_id=%s language=%s duration=%s chars=%s rms=%s gain=%s",
                user,
                target_user.id,
                getattr(info, "language", None),
                getattr(info, "duration", None),
                len(text),
                rms,
                round(applied_gain, 3),
            )
            return text or None
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            recognition_errors.append(message)
            logger.exception("Speech recognition process callback failed for user=%s", user)
            return None

    def text_cb(user, text):
        nonlocal transcript_text
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
            prior_text = transcript_text
            transcript_text = merge_transcript_text(transcript_text, text.strip())
            logger.info(
                "Merged speech transcript target_user_id=%s prior_chars=%s new_chars=%s merged_chars=%s",
                target_user.id,
                len(prior_text),
                len(text.strip()),
                len(transcript_text),
            )

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
        if recognition_errors and not transcript_text:
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

    transcript = transcript_text.strip()
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
reaction_stop_delay_ms = max(
    0,
    get_env_int("BOT_REACTION_STOP_DELAY_MS", DEFAULT_REACTION_STOP_DELAY_MS),
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
    for guild in client.guilds:
        try:
            bot_voice_channel = await fetch_bot_voice_channel(guild)
            if bot_voice_channel is not None:
                logger.info(
                    "Bot still appears connected in guild=%s channel=%s after restart; leaving server-side voice state untouched",
                    guild.id,
                    bot_voice_channel.id,
                )
        except Exception:
            logger.exception("Startup voice state check failed for guild %s", guild.id)
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
    if after.channel is None:
        stop_auto_listen(member.guild.id, disable=True)
        clear_conversation_history(member.guild.id)
        await remove_control_message_reaction_for_guild(member.guild)


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
        if message.guild and await force_leave_guild_voice(message.guild, reason="manual_leave"):
            await remove_control_message_reaction(message.channel)
            logger.info("Disconnected from voice in guild %s", message.guild.id)
            await message.channel.send("Left the voice channel.")
        else:
            await message.channel.send("I'm not in a voice channel right now.")
        return

    if command == "!clear":
        await handle_clear_command(message, content)
        return

    if command == "!talk" and len(parts) == 1:
        await enable_hands_free_mode(message)
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

        if message.guild and stop_auto_listen(message.guild.id, disable=True):
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
            await send_error(message.channel, "There is no active talk session, recording, or playback to stop.")
        return

    if command != "!say":
        return

    question = content[4:].strip()
    if not question:
        await message.channel.send("Usage: `!say your message here`")
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
        await handle_record_or_talk(
            context,
            duration_seconds,
            mode="talk",
            status_message="Recording and listening...",
        )
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
        "Stopping reaction-controlled recording guild=%s user=%s emoji=%s stop_delay_ms=%s",
        payload.guild_id,
        payload.user_id,
        payload.emoji,
        reaction_stop_delay_ms,
    )
    if reaction_stop_delay_ms > 0:
        await asyncio.sleep(reaction_stop_delay_ms / 1000)
    stop_active_recording(payload.guild_id)


if __name__ == "__main__":
    client.run(token, log_handler=None)
