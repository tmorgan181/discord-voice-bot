import asyncio
import array
import difflib
import io
import json
import logging
import math
import os
import random
import re
import sys
import uuid
import wave
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import time

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
DEFAULT_AUTO_LISTEN_MAX_LAUGH_TOKEN_RATIO = 0.6
DEFAULT_AUTO_LISTEN_MIN_NON_LAUGH_WORDS = 2
DEFAULT_AUDIO_GAIN = 1.15
DEFAULT_TARGET_RMS = 14000.0
DEFAULT_MAX_AUDIO_GAIN = 6.0
DEFAULT_WHISPER_MODEL = "base.en"
DEFAULT_WHISPER_BEAM_SIZE = 1
FFMPEG_SILENCE_INPUT = "anullsrc=r=48000:cl=stereo"
TWENTY_QUESTIONS_MAX_TURNS = 20
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
ACTIVE_TTS_PLAYBACKS: dict[int, dict] = {}
HANDS_FREE_PHASES: dict[int, str] = {}
PENDING_VOICE_EXIT_TASKS: dict[int, asyncio.Task] = {}
BONK_COUNTS: dict[int, int] = defaultdict(int)
ACTIVE_TWENTY_QUESTIONS: dict[int, dict] = {}
LAST_TTS_BUSY_NOTICE_AT: dict[int, float] = {}
LAST_PACKET_WARNING_AT: dict[tuple[str, int | None], float] = {}

COLOR_RESET = "\033[0m"
COLOR_DIM = "\033[2m"
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_MAGENTA = "\033[35m"
COLOR_CYAN = "\033[36m"

BARREL_ROLL_ASCII = """\
      __|__
--o--(_)--o--
    / ^ \\

           / v \\
      --o--(_)--o--
         __|__

             __|__//
         --o--(_)--o--
             / ^ \\
"""

PRESS_F_ASCII = """\
      ___________
     /           \\
    /   R.I.P.    \\
   /    Respect    \\
  /_________________\\
      ||       ||
      ||       ||
"""


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


def get_auto_listen_max_laugh_token_ratio() -> float:
    return max(
        0.0,
        min(1.0, get_env_float("BOT_AUTO_LISTEN_MAX_LAUGH_TOKEN_RATIO", DEFAULT_AUTO_LISTEN_MAX_LAUGH_TOKEN_RATIO)),
    )


def get_auto_listen_min_non_laugh_words() -> int:
    return max(
        0,
        get_env_int("BOT_AUTO_LISTEN_MIN_NON_LAUGH_WORDS", DEFAULT_AUTO_LISTEN_MIN_NON_LAUGH_WORDS),
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


def verbose_logging_enabled() -> bool:
    return env_flag("CHUNG_VERBOSE_LOGS", False) or "-v" in sys.argv or "--verbose" in sys.argv


class EssentialConsoleFilter(logging.Filter):
    SUPPRESSED_PREFIXES = (
        "Ensuring voice client",
        "Voice connect attempt",
        "Reusing existing voice connection",
        "Preparing TTS playback",
        "Generated TTS audio",
        "Stopping silent keepalive playback",
        "Started silent keepalive playback",
        "Reset conversation history",
        "Created conversation log",
        "Confirmed listener shutdown before TTS",
        "Rejected auto-listen audio chunk",
        "Started voice health monitor task",
        "Voice health guild=",
        "Added control reaction",
        "Removed control reaction",
        "Patched voice receive decoder",
        "Phase 3 dependencies ready",
        "Voice backend ready",
        "Runtime logging initialized",
    )

    def __init__(self, verbose: bool):
        super().__init__()
        self.verbose = verbose

    def filter(self, record: logging.LogRecord) -> bool:
        if self.verbose:
            return True
        if record.name != "chung":
            return False
        if record.levelno >= logging.WARNING:
            return True
        message = record.getMessage()
        return not any(message.startswith(prefix) for prefix in self.SUPPRESSED_PREFIXES)


class ConsoleFormatter(logging.Formatter):
    def __init__(self, verbose: bool):
        super().__init__()
        self.verbose = verbose

    def format(self, record: logging.LogRecord) -> str:
        if self.verbose:
            return f"{record.asctime if hasattr(record, 'asctime') else ''}{record.getMessage()}"

        message = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"{COLOR_RED}[ERROR]{COLOR_RESET} {message}"
        if record.levelno >= logging.WARNING:
            return f"{COLOR_YELLOW}[WARN]{COLOR_RESET} {message}"

        if record.name != "chung":
            return f"{COLOR_DIM}{message}{COLOR_RESET}"

        tag = "INFO"
        color = COLOR_CYAN
        if message.startswith("Logged in as"):
            tag, color = "READY", COLOR_GREEN
        elif message.startswith("Received message"):
            tag, color = "CMD", COLOR_BLUE
        elif message.startswith("Started auto-listen"):
            tag, color = "LISTEN", COLOR_GREEN
        elif message.startswith("Hands-free ready again"):
            tag, color = "LISTEN", COLOR_GREEN
        elif message.startswith("Auto-listen text"):
            tag, color = "HEARD", COLOR_CYAN
            parts = message.split(" text=", 1)
            if len(parts) == 2:
                message = parts[1]
        elif message.startswith("Ollama answered successfully"):
            tag, color = "THINK", COLOR_MAGENTA
        elif message.startswith("Started voice playback"):
            tag, color = "SPEAK", COLOR_MAGENTA
        elif message.startswith("Stop requested"):
            tag, color = "STOP", COLOR_YELLOW
        elif message.startswith("Disconnected from voice"):
            tag, color = "LEAVE", COLOR_YELLOW
        elif message.startswith("Connecting to voice channel"):
            tag, color = "JOIN", COLOR_GREEN

        return f"{color}[{tag}]{COLOR_RESET} {message}"


def setup_logging() -> logging.Logger:
    global SESSION_RUNTIME_LOG_PATH
    log_path = resolve_runtime_log_path()
    SESSION_RUNTIME_LOG_PATH = log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    verbose = verbose_logging_enabled()

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
    if verbose:
        stream_handler.setFormatter(formatter)
    else:
        stream_handler.setFormatter(ConsoleFormatter(verbose=False))
    stream_handler.addFilter(EssentialConsoleFilter(verbose=verbose))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    discord.utils.setup_logging(level=logging.INFO if verbose else logging.WARNING, root=False)

    # Keep the runtime log focused on the bot lifecycle instead of library chatter.
    reader_logger = logging.getLogger("discord.ext.voice_recv.reader")
    reader_logger.setLevel(logging.CRITICAL if not verbose else logging.INFO)
    reader_logger.disabled = not verbose
    logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING if not verbose else logging.INFO)
    logging.getLogger("discord.ext.voice_recv.opus").setLevel(logging.ERROR if not verbose else logging.WARNING)
    logging.getLogger("discord.player").setLevel(logging.WARNING if not verbose else logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING if not verbose else logging.INFO)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING if not verbose else logging.INFO)
    logging.getLogger("comtypes").setLevel(logging.WARNING if not verbose else logging.INFO)
    logging.getLogger("discord.client").setLevel(logging.WARNING if not verbose else logging.INFO)
    logging.getLogger("discord.gateway").setLevel(logging.WARNING if not verbose else logging.INFO)
    logging.getLogger("discord.voice_state").setLevel(logging.WARNING if not verbose else logging.INFO)

    logger = logging.getLogger("chung")
    logger.info("Runtime logging initialized at %s (verbose=%s)", log_path, verbose)
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


def should_log_packet_warning(kind: str, ssrc: int | None, *, interval_seconds: float = 10.0) -> bool:
    now = time.monotonic()
    key = (kind, ssrc)
    last_logged = LAST_PACKET_WARNING_AT.get(key, 0.0)
    if now - last_logged < interval_seconds:
        return False
    LAST_PACKET_WARNING_AT[key] = now
    return True


def patch_voice_recv_decoder() -> None:
    global VOICE_RECV_PATCHED
    if VOICE_RECV_PATCHED:
        return

    original_decode_packet = voice_recv_opus.PacketDecoder._decode_packet
    original_process_packet = voice_recv_opus.PacketDecoder._process_packet
    silence_frame = b"\x00" * voice_recv_opus.Decoder.SAMPLE_SIZE * voice_recv_opus.Decoder.SAMPLES_PER_FRAME

    def safe_decode_packet(self, packet):
        try:
            return original_decode_packet(self, packet)
        except OpusError as exc:
            ssrc = getattr(packet, "ssrc", None)
            if ssrc in ACTIVE_RECORDING_SSRCS:
                CORRUPT_PACKET_COUNTS[ssrc] = CORRUPT_PACKET_COUNTS.get(ssrc, 0) + 1
            elif should_log_packet_warning("opus", ssrc):
                logger.warning(
                    "Ignoring corrupt Opus packet outside active recording ssrc=%s sequence=%s error=%s",
                    ssrc,
                    getattr(packet, "sequence", None),
                    exc,
                )
            return packet, silence_frame

    def safe_process_packet(self, packet):
        try:
            return original_process_packet(self, packet)
        except ValueError as exc:
            if "Failed to decrypt" not in str(exc):
                raise
            ssrc = getattr(packet, "ssrc", None)
            if should_log_packet_warning("decrypt", ssrc):
                logger.warning(
                    "Dropping undecryptable voice packet ssrc=%s sequence=%s timestamp=%s error=%s",
                    ssrc,
                    getattr(packet, "sequence", None),
                    getattr(packet, "timestamp", None),
                    exc,
                )
            if packet:
                self._last_seq = packet.sequence
                self._last_ts = packet.timestamp
            return None

    voice_recv_opus.PacketDecoder._decode_packet = safe_decode_packet
    voice_recv_opus.PacketDecoder._process_packet = safe_process_packet
    VOICE_RECV_PATCHED = True
    logger.info("Patched voice receive decoder to tolerate corrupt Opus packets and decrypt failures")


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


def set_hands_free_phase(guild_id: int, phase: str) -> None:
    HANDS_FREE_PHASES[guild_id] = phase
    logger.info("Hands-free phase guild=%s phase=%s", guild_id, phase)


def clear_hands_free_phase(guild_id: int) -> None:
    if HANDS_FREE_PHASES.pop(guild_id, None) is not None:
        logger.info("Cleared hands-free phase guild=%s", guild_id)


def get_hands_free_phase(guild_id: int) -> str:
    return HANDS_FREE_PHASES.get(guild_id, "idle")


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


def build_twenty_questions_prompt(game: dict) -> str:
    history = game.get("history", [])
    turns_used = game.get("turns_used", 0)
    remaining = max(0, TWENTY_QUESTIONS_MAX_TURNS - turns_used)
    lines = [
        "You are playing 20 Questions.",
        "The user is thinking of a secret word or phrase.",
        "You must ask concise yes/no questions, or make a direct guess when confident.",
        f"You have {remaining} turns remaining out of {TWENTY_QUESTIONS_MAX_TURNS}.",
        "Reply in exactly one of these formats:",
        "QUESTION: <short yes/no question>",
        "GUESS: <your guess>",
        "Keep questions under 16 words and guesses under 10 words.",
        "Do not include explanation or any extra text.",
    ]
    if history:
        lines.append("Game history:")
        for item in history[-40:]:
            lines.append(f"{item['role']}: {item['content']}")
    else:
        lines.append("This is the first turn. Start broad and smart.")
    if remaining <= 1:
        lines.append("You should make a guess now.")
    return "\n".join(lines)


def ask_ollama_for_twenty_questions_turn(game: dict) -> tuple[str, str]:
    response = ask_ollama(build_twenty_questions_prompt(game))
    normalized = response.strip()
    if normalized.upper().startswith("QUESTION:"):
        content = normalized.split(":", 1)[1].strip()
        return "question", content or "Is it a physical object?"
    if normalized.upper().startswith("GUESS:"):
        content = normalized.split(":", 1)[1].strip()
        return "guess", content or "a rock"
    if "?" in normalized:
        return "question", normalized
    return "guess", normalized or "a rock"


def normalize_yes_no_answer(text: str) -> str | None:
    raw_lowered = text.lower().strip()
    if "don't know" in raw_lowered or "dont know" in raw_lowered or "not sure" in raw_lowered:
        return "unknown"

    lowered = re.sub(r"[^a-z\s]", " ", raw_lowered)
    tokens = [token for token in lowered.split() if token]
    if not tokens:
        return None

    positive = {"yes", "yep", "yeah", "yup", "correct", "right", "true", "sure", "affirmative"}
    negative = {"no", "nope", "nah", "false", "incorrect", "wrong", "negative"}
    maybe = {"maybe", "sometimes", "sorta", "kind", "kindof", "depends", "partly", "partially"}
    unknown = {"unknown", "idk", "unsure"}

    if any(token in positive for token in tokens):
        return "yes"
    if any(token in negative for token in tokens):
        return "no"
    if any(token in maybe for token in tokens):
        return "maybe"
    if "i" in tokens and "know" in tokens:
        return "unknown"
    if any(token in unknown for token in tokens):
        return "unknown"
    return None


def is_twenty_questions_start_request(text: str) -> bool:
    lowered = text.strip().lower()
    return bool(
        re.search(
            r"\b(?:20|twenty)\s+questions?\b",
            lowered,
        )
    )


def is_twenty_questions_end_request(text: str) -> bool:
    lowered = text.strip().lower()
    return bool(
        re.search(
            r"\b(?:" 
            r"end(?:\s+of)?\s+(?:the\s+)?(?:game|again)"
            r"|and\s+(?:the\s+)?game"
            r"|stop\s+(?:the\s+)?game"
            r"|quit\s+(?:the\s+)?game"
            r")\b",
            lowered,
        )
    )


def is_twenty_questions_stop_command(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered in {
        "!20q stop",
        "!20q end",
        "!20q quit",
        "!stop",
        "!endgame",
        "!stopgame",
    }


async def advance_twenty_questions_turn(context, game: dict) -> None:
    turn_kind, content = await asyncio.to_thread(ask_ollama_for_twenty_questions_turn, game)
    game["turns_used"] = game.get("turns_used", 0) + 1
    game["awaiting"] = "guess_result" if turn_kind == "guess" else "answer"
    game["last_turn_kind"] = turn_kind
    game["last_bot_content"] = content
    game.setdefault("history", []).append(
        {
            "role": "assistant",
            "content": f"{turn_kind.upper()}: {content}",
        }
    )

    if turn_kind == "guess":
        reply = f"Guess {game['turns_used']}/{TWENTY_QUESTIONS_MAX_TURNS}: is it `{content}`?"
        spoken = f"Guess {game['turns_used']}. Is it {content}?"
    else:
        reply = f"Question {game['turns_used']}/{TWENTY_QUESTIONS_MAX_TURNS}: {content}"
        spoken = content
    await send_fun_response(context, reply, speak_text_reply=spoken)


async def start_twenty_questions(context) -> None:
    guild = getattr(context, "guild", None)
    if guild is None:
        await send_error(context.channel, "20 Questions only works inside a server.")
        return
    existing = ACTIVE_TWENTY_QUESTIONS.get(guild.id)
    if existing and existing.get("channel_id") == context.channel.id:
        await context.channel.send("20 Questions is already running here. Use `!20q status` or `!20q stop`.")
        return

    ACTIVE_TWENTY_QUESTIONS[guild.id] = {
        "channel_id": context.channel.id,
        "host_user_id": getattr(getattr(context, "author", None), "id", None),
        "turns_used": 0,
        "awaiting": "answer",
        "last_turn_kind": None,
        "last_bot_content": None,
        "history": [
            {
                "role": "system",
                "content": "The player has started a new game and is thinking of a word or phrase.",
            }
        ],
    }
    await context.channel.send(
        "Think of a word or phrase. Answer with `yes`, `no`, `maybe`, or `don't know`, and I'll try to get it in 20."
    )
    await advance_twenty_questions_turn(context, ACTIVE_TWENTY_QUESTIONS[guild.id])


async def stop_twenty_questions(context, *, message: str | None = None) -> bool:
    guild = getattr(context, "guild", None)
    if guild is None:
        return False
    game = ACTIVE_TWENTY_QUESTIONS.pop(guild.id, None)
    if game is None:
        return False
    if message:
        await context.channel.send(message)
    return True


async def maybe_handle_twenty_questions_reply(context, content: str) -> bool:
    guild = getattr(context, "guild", None)
    if guild is None:
        return False
    game = ACTIVE_TWENTY_QUESTIONS.get(guild.id)
    if game is None or game.get("channel_id") != context.channel.id:
        return False

    stripped = content.strip()
    if not stripped:
        return True

    lowered = stripped.lower()
    if is_twenty_questions_stop_command(lowered) or is_twenty_questions_end_request(lowered):
        await stop_twenty_questions(context, message="20 Questions ended.")
        return True

    normalized = normalize_yes_no_answer(stripped)
    if normalized is None:
        await context.channel.send("20 Questions wants `yes`, `no`, `maybe`, or `don't know`.")
        return True

    last_turn_kind = game.get("last_turn_kind")
    last_bot_content = game.get("last_bot_content") or ""
    if last_turn_kind == "guess":
        game.setdefault("history", []).append({"role": "user", "content": f"GUESS RESULT: {normalized}"})
        if normalized == "yes":
            turns_used = game.get("turns_used", 0)
            await stop_twenty_questions(
                context,
                message=f"I win in {turns_used} turns. Suspiciously easy, honestly.",
            )
            return True
        game.setdefault("history", []).append(
            {"role": "system", "content": f"The guess `{last_bot_content}` was wrong."}
        )
    else:
        game.setdefault("history", []).append(
            {"role": "user", "content": f"ANSWER to `{last_bot_content}`: {normalized}"}
        )

    if game.get("turns_used", 0) >= TWENTY_QUESTIONS_MAX_TURNS:
        await stop_twenty_questions(context, message="I'm out of turns. You win this round.")
        return True

    await advance_twenty_questions_turn(context, game)
    return True


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
    if await maybe_handle_fun_trigger(context, transcript):
        await ensure_auto_listen(context.guild, context.channel)
        return
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


def build_wav_bytes_from_pcm(
    frames: bytes,
    *,
    channels: int = 2,
    sample_width: int = 2,
    sample_rate: int = 48000,
) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)
    return buffer.getvalue()


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


def normalize_word_token(token: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", token.lower())


def is_laughter_token(token: str) -> bool:
    normalized = normalize_word_token(token)
    if normalized in {"lol", "lmao", "lmfao", "rofl", "haha", "hehe", "hihi", "hoho", "hahae"}:
        return True
    if re.fullmatch(r"(ha|he|hi|ho|hu){2,}", normalized):
        return True
    return False


def analyze_transcript_tokens(text: str) -> dict:
    tokens = [token for token in re.findall(r"[a-z0-9']+", text.lower()) if len(token) >= 2]
    laugh_tokens = [token for token in tokens if is_laughter_token(token)]
    non_laugh_tokens = [token for token in tokens if not is_laughter_token(token)]
    laugh_ratio = (len(laugh_tokens) / len(tokens)) if tokens else 0.0
    return {
        "tokens": tokens,
        "laugh_tokens": laugh_tokens,
        "non_laugh_tokens": non_laugh_tokens,
        "laugh_ratio": laugh_ratio,
        "distinct_non_laugh_words": len(set(non_laugh_tokens)),
    }


def should_ignore_laughter_transcript(text: str) -> tuple[bool, dict]:
    stats = analyze_transcript_tokens(text)
    laugh_ratio = stats["laugh_ratio"]
    distinct_non_laugh_words = stats["distinct_non_laugh_words"]
    max_laugh_ratio = get_auto_listen_max_laugh_token_ratio()
    min_non_laugh_words = get_auto_listen_min_non_laugh_words()

    ignored = (
        bool(stats["tokens"])
        and laugh_ratio >= max_laugh_ratio
        and distinct_non_laugh_words < min_non_laugh_words
    )
    stats["laugh_ratio"] = round(laugh_ratio, 3)
    stats["max_laugh_ratio"] = max_laugh_ratio
    stats["min_non_laugh_words"] = min_non_laugh_words
    return ignored, stats


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


async def maybe_reject_message_during_tts(message: discord.Message, command: str) -> bool:
    guild = message.guild
    if guild is None:
        return False

    playback = ACTIVE_TTS_PLAYBACKS.get(guild.id)
    if not playback:
        return False

    voice_client = playback.get("voice_client")
    if not isinstance(voice_client, discord.VoiceClient):
        return False
    if not voice_client.is_connected():
        return False
    if is_idle_keepalive_active(voice_client):
        return False

    allowed_commands = {"!stop", "!leave", "!endtalk", "!stoptalk", "!talkoff"}
    if command in allowed_commands:
        return False

    now = time.monotonic()
    last_notice = LAST_TTS_BUSY_NOTICE_AT.get(guild.id, 0.0)
    if now - last_notice >= 2.0:
        LAST_TTS_BUSY_NOTICE_AT[guild.id] = now
        await message.channel.send("I'm talking right now, so I'm ignoring new messages until the TTS finishes.")
    return True


async def send_fun_response(
    context,
    text: str,
    *,
    speak_text_reply: str | None = None,
) -> None:
    await context.channel.send(text)
    spoken = speak_text_reply.strip() if speak_text_reply else ""
    if not spoken:
        return

    guild = getattr(context, "guild", None)
    author = getattr(context, "author", None)
    current_voice_client = get_current_voice_client(guild) if guild is not None else None
    has_active_voice = current_voice_client is not None
    has_configured_voice = bool(get_env_value("BOT_VOICE_CHANNEL_ID"))
    author_in_voice = isinstance(author, discord.Member) and author.voice is not None
    if not (has_active_voice or has_configured_voice or author_in_voice):
        return

    try:
        voice_client = current_voice_client
        if voice_client is None or not voice_client.is_connected():
            voice_client = await ensure_voice_client(context)
        if voice_client is not None:
            await speak_text(voice_client, spoken)
    except discord.ClientException:
        logger.exception("Discord voice error during fun trigger playback")
        if guild is not None:
            try:
                await reset_voice_client(guild)
                voice_client = await ensure_voice_client(context)
                if voice_client is not None:
                    await speak_text(voice_client, spoken)
                    return
            except Exception:
                logger.exception("Voice recovery retry failed during fun trigger playback")
    except Exception:
        logger.exception("TTS playback failed during fun trigger")
        if guild is not None:
            try:
                await reset_voice_client(guild)
                voice_client = await ensure_voice_client(context)
                if voice_client is not None:
                    await speak_text(voice_client, spoken)
                    return
            except Exception:
                logger.exception("Voice recovery retry failed after TTS playback error")


def build_enhance_text(content: str) -> str:
    del content

    subject = random.choice(
        [
            "the suspicious pixel",
            "the blurry evidence",
            "the mystery blob",
            "one deeply unhelpful shadow",
            "the least trustworthy jpeg in the room",
            "an emotionally compromised screenshot",
            "the crunchy artifact near the crime scene",
            "whatever that glowing smudge is",
        ]
    )

    first_percent = random.randint(18, 74)
    second_percent = random.randint(7, 41)
    first_quality = random.choice(
        [
            "dramatic",
            "suspicious",
            "cinematic",
            "concerning",
            "forensically spicy",
            "unnecessarily intense",
        ]
    )
    second_quality = random.choice(
        [
            "legally questionable",
            "haunted",
            "goblin-adjacent",
            "overengineered",
            "deeply cursed",
            "evidence-shaped",
        ]
    )
    ending = random.choice(
        [
            "and somehow still blurry.",
            "yet remains spiritually low-resolution.",
            "and absolutely not admissible anywhere.",
            "but the vibes are now significantly worse.",
            "and the pixel has learned nothing.",
        ]
    )

    return (
        "ENHANCING...\n"
        f"Target: `{subject[:80]}`\n"
        f"Result: {first_percent}% more {first_quality}, {second_percent}% more {second_quality}, {ending}"
    )


def resolve_credits_cast(context) -> list[tuple[int | None, str]]:
    members: list[tuple[int | None, str]] = []
    guild = getattr(context, "guild", None)
    author = getattr(context, "author", None)
    voice_state = getattr(author, "voice", None)
    voice_channel = getattr(voice_state, "channel", None)

    if voice_channel is None and guild is not None:
        current_voice = get_current_voice_client(guild)
        if current_voice is not None:
            voice_channel = current_voice.channel

    if voice_channel is not None:
        for member in voice_channel.members:
            if member.bot:
                continue
            members.append((member.id, member.display_name))

    if not members and author is not None:
        display_name = getattr(author, "display_name", str(author))
        members.append((getattr(author, "id", None), display_name))

    return members[:8]


def build_roll_credits_text(context) -> str:
    roles = [
        "Executive Producer of Bad Ideas",
        "Lead Barrel Roll Coordinator",
        "Bonk Safety Inspector",
        "Chief Blur Enhancement Officer",
        "Respect Operations Manager",
        "Goblin Containment Specialist",
        "Assistant to the Chaos Director",
        "Whee Compliance Auditor",
    ]
    taglines = [
        "No budgets were harmed in the making of this session.",
        "Filmed before a live studio goblin.",
        "All stunts performed by emotionally underqualified professionals.",
        "No one was prepared, least of all the narrator.",
        "Respect was paid. Repeatedly.",
        "Any resemblance to competent planning is purely coincidental.",
    ]
    cast = resolve_credits_cast(context)
    randomized_roles = roles[:]
    random.shuffle(randomized_roles)
    credit_lines = ["## Roll Credits", ""]
    for index, (member_id, name) in enumerate(cast):
        role = randomized_roles[index % len(randomized_roles)]
        bonk_total = BONK_COUNTS.get(member_id, 0) if member_id is not None else 0
        bonk_suffix = f" | Bonks: {bonk_total}" if bonk_total else ""
        credit_lines.append(f"{name} as {role}{bonk_suffix}")
    credit_lines.append("")
    credit_lines.append(random.choice(taglines))
    return "\n".join(credit_lines)


def matches_fun_trigger(content: str) -> bool:
    lowered = content.strip().lower()
    normalized_single_word = re.sub(r"^[^\w]+|[^\w]+$", "", lowered)
    return any(
        (
            bool(re.search(r"\bbarrel\b", lowered)),
            bool(re.search(r"\b(?:press\s+f|f|pay\s+respects?|respects?)\b", lowered)),
            bool(re.search(r"\b(?:bonk|vonk|wonk)\b", lowered)) or normalized_single_word in {"back", "bank"},
            bool(re.search(r"\benhance(d)?\b", lowered)),
            "roll credits" in lowered or "roll credit" in lowered or bool(re.search(r"\bcredits?\b", lowered)),
        )
    )


def resolve_bonk_target(context, content: str):
    author = getattr(context, "author", None)
    guild = getattr(context, "guild", None)
    mentions = getattr(context, "mentions", None) or []

    if mentions:
        target = mentions[0]
        return getattr(target, "id", None), getattr(target, "display_name", str(target))

    match = re.search(r"\b(?:bonk|vonk|wonk|back|bank)\b[\s,:-]*(.+)", content, flags=re.IGNORECASE)
    if not match:
        return getattr(author, "id", None), getattr(author, "display_name", str(author) if author else "someone")

    raw_target = match.group(1).strip()
    raw_target = re.sub(r"^[\"'`]+|[\"'`]+$", "", raw_target).strip()
    raw_target = re.sub(r"^<@!?(\d+)>$", r"\1", raw_target)
    if not raw_target:
        return getattr(author, "id", None), getattr(author, "display_name", str(author) if author else "someone")

    if guild is not None:
        lowered_target = raw_target.casefold()
        candidates: list[tuple[discord.Member, list[str]]] = []
        for member in guild.members:
            if member.bot:
                continue
            names = [
                str(member.id),
                member.display_name.casefold(),
                member.name.casefold(),
                str(member).casefold(),
                getattr(member, "global_name", "") or "",
                getattr(member, "nick", "") or "",
            ]
            normalized_names = [name.casefold() for name in names if name]
            candidates.append((member, normalized_names))

        for member, names in candidates:
            if lowered_target in names:
                return member.id, member.display_name

        partial_matches: list[tuple[int, discord.Member]] = []
        for member, names in candidates:
            best_partial_score = 0
            for name in names:
                if lowered_target in name:
                    best_partial_score = max(best_partial_score, len(lowered_target) * 100)
                elif name in lowered_target:
                    best_partial_score = max(best_partial_score, len(name) * 10)
            if best_partial_score:
                partial_matches.append((best_partial_score, member))

        if partial_matches:
            partial_matches.sort(key=lambda item: item[0], reverse=True)
            best_member = partial_matches[0][1]
            return best_member.id, best_member.display_name

        fuzzy_matches: list[tuple[float, discord.Member]] = []
        for member, names in candidates:
            best_ratio = max((difflib.SequenceMatcher(None, lowered_target, name).ratio() for name in names), default=0.0)
            if best_ratio >= 0.45:
                fuzzy_matches.append((best_ratio, member))

        if fuzzy_matches:
            fuzzy_matches.sort(key=lambda item: item[0], reverse=True)
            best_member = fuzzy_matches[0][1]
            return best_member.id, best_member.display_name

    return getattr(author, "id", None), getattr(author, "display_name", str(author) if author else "someone")


async def maybe_handle_fun_trigger(context, content: str) -> bool:
    stripped = content.strip()
    lowered = stripped.lower()
    normalized_single_word = re.sub(r"^[^\w]+|[^\w]+$", "", lowered)
    author = getattr(context, "author", None)

    if re.search(r"\bbarrel\b", lowered):
        await send_fun_response(
            context,
            f"```text\n{BARREL_ROLL_ASCII}\n```\nwheeeee",
            speak_text_reply="Wheeeee eee eee eee.",
        )
        return True

    if re.search(r"\b(?:press\s+f|f|pay\s+respects?|respects?)\b", lowered):
        await send_fun_response(
            context,
            f"```text\n{PRESS_F_ASCII}\n```\nrespect has been paid",
            speak_text_reply="respect has been paid",
        )
        return True

    if re.search(r"\b(?:bonk|vonk|wonk)\b", lowered) or normalized_single_word in {"back", "bank"}:
        target_id, target_name = resolve_bonk_target(context, stripped)
        if target_id is not None:
            BONK_COUNTS[target_id] += 1
            bonk_total = BONK_COUNTS[target_id]
        else:
            bonk_total = 1
        bonk_message = f"*bonk* `{target_name}` has been bonked {bonk_total} time{'s' if bonk_total != 1 else ''}."
        await send_fun_response(
            context,
            bonk_message,
            speak_text_reply=f"bonk. {target_name} has been bonked {bonk_total} time{'s' if bonk_total != 1 else ''}.",
        )
        return True

    if re.search(r"\benhance(d)?\b", lowered):
        await send_fun_response(
            context,
            build_enhance_text(stripped),
            speak_text_reply="enhancing. results remain extremely suspicious.",
        )
        return True

    if "roll credits" in lowered or "roll credit" in lowered or re.search(r"\bcredits?\b", lowered):
        credits_text = build_roll_credits_text(context)
        await send_fun_response(
            context,
            credits_text,
            speak_text_reply=credits_text.replace("## ", "").replace("`", ""),
        )
        return True

    return False


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


async def finalize_voice_exit_if_still_disconnected(guild: discord.Guild, expected_before_channel_id: int | None) -> None:
    try:
        await asyncio.sleep(1.0)
        bot_voice_channel = await fetch_bot_voice_channel(guild)
        if bot_voice_channel is not None:
            logger.info(
                "Ignoring transient voice exit for guild=%s expected_before_channel=%s current_channel=%s",
                guild.id,
                expected_before_channel_id,
                bot_voice_channel.id,
            )
            return
        logger.info(
            "Confirmed bot voice exit guild=%s expected_before_channel=%s; clearing session state",
            guild.id,
            expected_before_channel_id,
        )
        stop_auto_listen(guild.id, disable=True)
        clear_conversation_history(guild.id)
        await remove_control_message_reaction_for_guild(guild)
    finally:
        PENDING_VOICE_EXIT_TASKS.pop(guild.id, None)


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
    if isinstance(voice_client, voice_recv.VoiceRecvClient) and voice_client.is_listening():
        logger.info(
            "Skipping idle keepalive because voice receive is active in channel %s",
            voice_client.channel.id if voice_client.channel else None,
        )
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
        if hasattr(voice_client, "stop_playing"):
            voice_client.stop_playing()
        else:
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

    target_channel: discord.VoiceChannel | None = None

    if isinstance(message.author, discord.Member) and message.author.voice:
        target_channel = message.author.voice.channel

    if target_channel is None:
        configured_voice_channel_id = get_env_value("BOT_VOICE_CHANNEL_ID")
        if configured_voice_channel_id:
            channel = message.guild.get_channel(int(configured_voice_channel_id))
            if channel is None:
                try:
                    channel = await client.fetch_channel(int(configured_voice_channel_id))
                except Exception:
                    logger.exception("Could not fetch configured voice channel id=%s", configured_voice_channel_id)
            if not isinstance(channel, discord.VoiceChannel):
                await send_error(
                    message.channel,
                    "The configured `BOT_VOICE_CHANNEL_ID` is not a valid voice channel.",
                )
                return None
            target_channel = channel

    if target_channel is None:
        await send_error(message.channel, "Join a voice channel first, then try again.")
        return None

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
        logger.info("TTS playback ended guild=%s error=%s", guild_id, error)
        if guild_id is not None:
            ACTIVE_TTS_PLAYBACKS.pop(guild_id, None)
        if error:
            logger.exception("Voice playback error", exc_info=error)
        setattr(voice_client, "_idle_keepalive_active", False)
        try:
            audio_path.unlink(missing_ok=True)
        except PermissionError:
            logger.warning("Temporary TTS file still in use during cleanup: %s", audio_path)
        except OSError:
            logger.warning("Could not delete temporary audio file %s", audio_path)
        fallback_channel = None
        if guild is not None:
            session = AUTO_LISTEN_SESSIONS.get(guild.id)
            fallback_channel = session.get("text_channel") if session else None
        if voice_client.is_connected():
            if guild is not None and guild.id in AUTO_LISTEN_ENABLED_GUILDS:
                try:
                    asyncio.run_coroutine_threadsafe(
                        resume_hands_free_after_playback(guild, fallback_channel),
                        client.loop,
                    )
                except Exception:
                    logger.exception("Failed to resume auto-listen after TTS playback")
            else:
                try:
                    start_idle_keepalive(voice_client)
                except Exception:
                    logger.exception("Failed to restart idle keepalive after TTS playback")

    if guild_id is not None:
        set_hands_free_phase(guild_id, "playing")
        stop_auto_listen(guild_id)
        stop_voice_capture(voice_client)
        if isinstance(voice_client, voice_recv.VoiceRecvClient):
            listener_stopped = await wait_for_voice_listener_stop(voice_client)
            logger.info(
                "Confirmed listener shutdown before TTS guild=%s stopped=%s",
                guild_id,
                listener_stopped,
            )
    if voice_client.is_playing():
        stop_idle_keepalive(voice_client)

    if guild_id is not None:
        ACTIVE_TTS_PLAYBACKS[guild_id] = {
            "voice_client": voice_client,
            "text": text,
        }

    source = discord.FFmpegPCMAudio(str(audio_path))
    try:
        voice_client.play(source, after=cleanup)
    except Exception:
        if guild_id is not None:
            ACTIVE_TTS_PLAYBACKS.pop(guild_id, None)
        try:
            audio_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete temporary audio file %s after playback start failure", audio_path)
        raise
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
    if fallback_channel is not None:
        return fallback_channel
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
    return None


def stop_auto_listen(guild_id: int, *, disable: bool = False) -> bool:
    if disable:
        AUTO_LISTEN_ENABLED_GUILDS.discard(guild_id)
        clear_hands_free_phase(guild_id)
    session = AUTO_LISTEN_SESSIONS.pop(guild_id, None)
    if not session:
        return False

    for handle_key in ("finalize_handle",):
        handle = session.get(handle_key)
        if handle is not None:
            handle.cancel()

    voice_client = session.get("voice_client")
    if isinstance(voice_client, voice_recv.VoiceRecvClient) and voice_client.is_listening():
        logger.info("Stopping auto-listen for guild=%s", guild_id)
        voice_client.stop_listening()
    return True


async def wait_for_voice_listener_stop(
    voice_client: voice_recv.VoiceRecvClient,
    *,
    timeout_seconds: float = 1.0,
) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while voice_client.is_listening() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    return not voice_client.is_listening()


async def handle_auto_transcript(guild_id: int, user, text: str) -> None:
    phase = get_hands_free_phase(guild_id)
    if phase != "listening":
        logger.warning(
            "[HANDS-FREE] Dropping transcript because phase is not listening guild=%s user=%s phase=%s text=%s",
            guild_id,
            getattr(user, "id", None),
            phase,
            text,
        )
        return
    session = AUTO_LISTEN_SESSIONS.get(guild_id)
    if not session or not text.strip():
        return
    await process_auto_transcript(session["guild"], user, text, session.get("text_channel"))


async def process_auto_transcript(
    guild: discord.Guild,
    user,
    text: str,
    fallback_channel=None,
) -> None:
    stripped_text = text.strip()

    async def resume_if_needed() -> None:
        await resume_hands_free_if_enabled(guild, fallback_channel)

    if not stripped_text:
        await resume_if_needed()
        return

    if client.user and user.id == client.user.id:
        await resume_if_needed()
        return

    session = AUTO_LISTEN_SESSIONS.get(guild.id)
    last_text = session.get("last_text", "") if session else ""
    if stripped_text.lower() == str(last_text).strip().lower():
        logger.info("Ignoring duplicate auto transcript guild=%s user=%s text=%s", guild.id, user.id, text)
        await resume_if_needed()
        return

    is_fun_trigger = matches_fun_trigger(stripped_text)
    if not is_fun_trigger:
        distinct_words = extract_distinct_words(text)
        min_distinct_words = get_auto_listen_min_distinct_words()
        if len(distinct_words) < min_distinct_words:
            logger.info(
                "Ignoring short auto transcript guild=%s user=%s distinct_words=%s min_required=%s text=%s",
                guild.id,
                user.id,
                len(distinct_words),
                min_distinct_words,
                text,
            )
            await resume_if_needed()
            return

    ignore_laughter, laughter_stats = should_ignore_laughter_transcript(text)
    if ignore_laughter:
        logger.info(
            "Ignoring laughter-like auto transcript guild=%s user=%s stats=%s text=%s",
            guild.id,
            user.id,
            laughter_stats,
            text,
        )
        await resume_if_needed()
        return

    if session and session.get("busy"):
        logger.info("Ignoring auto transcript while another auto response is in flight guild=%s user=%s", guild.id, user.id)
        await resume_if_needed()
        return

    channel = await resolve_text_channel_for_guild(guild, fallback_channel)
    if channel is None:
        logger.warning("Auto-listen could not resolve a text channel for guild=%s", guild.id)
        await resume_if_needed()
        return

    if session is not None:
        session["busy"] = True
        session["last_text"] = stripped_text
    set_hands_free_phase(guild.id, "processing")
    stop_auto_listen(guild.id)
    try:
        await channel.send(f"Heard {getattr(user, 'display_name', user)}: {stripped_text}")
        message_like = SimpleNamespace(guild=guild, channel=channel, author=user)
        if is_twenty_questions_start_request(stripped_text):
            await start_twenty_questions(message_like)
            return
        if is_twenty_questions_end_request(stripped_text):
            if not await stop_twenty_questions(message_like, message="20 Questions ended."):
                await channel.send("There isn't an active 20 Questions game right now.")
            await resume_if_needed()
            return
        if await maybe_handle_twenty_questions_reply(message_like, stripped_text):
            voice_client = get_current_voice_client(guild)
            if not (voice_client and voice_client.is_connected() and voice_client.is_playing()):
                await resume_if_needed()
            return
        if await maybe_handle_fun_trigger(message_like, stripped_text):
            return
        await respond_with_ollama(
            message_like,
            stripped_text,
            speak_reply=True,
            log_source="auto",
        )
    finally:
        current = AUTO_LISTEN_SESSIONS.get(guild.id)
        if current is not None:
            current["busy"] = False


async def ensure_auto_listen(guild: discord.Guild | None, fallback_channel=None) -> None:
    if guild is not None and guild.id in AUTO_LISTEN_ENABLED_GUILDS:
        set_hands_free_phase(guild.id, "listening")
    await ensure_speech_listener(guild, fallback_channel)


async def resume_hands_free_after_playback(guild: discord.Guild, fallback_channel=None) -> None:
    stop_auto_listen(guild.id)
    await ensure_auto_listen(guild, fallback_channel)
    voice_client = get_current_voice_client(guild)
    if (
        voice_client
        and voice_client.is_connected()
        and not voice_client.is_playing()
        and not (isinstance(voice_client, voice_recv.VoiceRecvClient) and voice_client.is_listening())
    ):
        try:
            start_idle_keepalive(voice_client)
        except Exception:
            logger.exception("Failed to restart idle keepalive after resuming hands-free mode")
    logger.info(
        "Hands-free ready again guild=%s listening=%s playing=%s idle_keepalive=%s",
        guild.id,
        isinstance(voice_client, voice_recv.VoiceRecvClient) and voice_client.is_listening() if voice_client else False,
        voice_client.is_playing() if voice_client else False,
        is_idle_keepalive_active(voice_client),
    )


async def resume_hands_free_if_enabled(guild: discord.Guild, fallback_channel=None) -> None:
    if guild.id not in AUTO_LISTEN_ENABLED_GUILDS:
        return
    await resume_hands_free_after_playback(guild, fallback_channel)




async def ensure_speech_listener(
    guild: discord.Guild | None,
    fallback_channel=None,
) -> None:
    if guild is None:
        logger.info("Skipping auto-listen because guild is None")
        return
    if guild.id not in AUTO_LISTEN_ENABLED_GUILDS:
        logger.info("Skipping auto-listen for guild=%s because hands-free mode is disabled", guild.id)
        return
    if guild.id in ACTIVE_RECORDINGS:
        logger.info("Skipping auto-listen for guild=%s because an active recording is running", guild.id)
        return

    voice_client = get_current_voice_client(guild)
    if not isinstance(voice_client, voice_recv.VoiceRecvClient) or not voice_client.is_connected():
        logger.info("Skipping auto-listen for guild=%s because no connected voice recv client is available", guild.id)
        return
    if guild.id in ACTIVE_TTS_PLAYBACKS:
        logger.info("Skipping auto-listen for guild=%s because TTS playback is still active", guild.id)
        return
    if get_hands_free_phase(guild.id) != "listening":
        logger.info(
            "Skipping auto-listen for guild=%s because hands-free phase is %s",
            guild.id,
            get_hands_free_phase(guild.id),
        )
        return

    existing = AUTO_LISTEN_SESSIONS.get(guild.id)
    if (
        existing
        and existing.get("voice_client") is voice_client
        and voice_client.is_listening()
    ):
        if fallback_channel is not None:
            existing["text_channel"] = fallback_channel
        return

    stop_auto_listen(guild.id)
    if voice_client.is_listening():
        stopped = await wait_for_voice_listener_stop(voice_client)
        logger.info("Waited for prior listener shutdown guild=%s stopped=%s", guild.id, stopped)
        if not stopped:
            return
    text_channel = await resolve_text_channel_for_guild(guild, fallback_channel)

    pause_threshold = get_auto_listen_silence_seconds()
    phrase_limit = get_auto_listen_phrase_limit_seconds()
    listener_token = uuid.uuid4().hex

    def process_cb(recognizer, audio, user):
        try:
            if user is None or (client.user and user.id == client.user.id):
                return None
            listener_age = round(time.monotonic() - listener_started_at, 3)
            audio_data = audio.get_wav_data()
            accepted, reason, stats = should_accept_auto_audio_chunk(audio_data)
            duration_s = stats.get("duration_seconds", 0)
            if duration_s > listener_age:
                stale_s = duration_s - listener_age
                logger.info(
                    "Trimming stale audio from chunk guild=%s user=%s stale_s=%s duration_s=%s listener_age_s=%s",
                    guild.id,
                    user.id,
                    round(stale_s, 3),
                    duration_s,
                    listener_age,
                )
                with wave.open(io.BytesIO(audio_data), "rb") as wav_in:
                    params = wav_in.getparams()
                    frames = wav_in.readframes(wav_in.getnframes())
                stale_bytes = int(stale_s * params.framerate) * params.nchannels * params.sampwidth
                trimmed = frames[stale_bytes:]
                if not trimmed:
                    return None
                audio_data = build_wav_bytes_from_pcm(
                    trimmed,
                    channels=params.nchannels,
                    sample_width=params.sampwidth,
                    sample_rate=params.framerate,
                )
                accepted, reason, stats = should_accept_auto_audio_chunk(audio_data)
            if not accepted:
                logger.info(
                    "Rejected auto-listen audio chunk guild=%s user=%s reason=%s stats=%s listener_age_s=%s",
                    guild.id,
                    user.id,
                    reason,
                    stats,
                    listener_age,
                )
                return None
            text, info, rms, applied_gain = transcribe_wav_bytes(audio_data)
            logger.info(
                "Auto-listen speech callback completed guild=%s user=%s language=%s duration=%s chars=%s rms=%s gain=%s listener_age_s=%s",
                guild.id,
                user.id,
                getattr(info, "language", None),
                getattr(info, "duration", None),
                len(text),
                rms,
                round(applied_gain, 3),
                listener_age,
            )
            return text or None
        except Exception:
            logger.exception("Auto-listen speech callback failed guild=%s user=%s", guild.id, getattr(user, "id", None))
            return None

    def text_cb(user, text):
        if user is None or not text or not text.strip():
            return
        phase = get_hands_free_phase(guild.id)
        if phase != "listening":
            logger.warning(
                "[HANDS-FREE] Ignoring queued transcript because phase is not listening guild=%s user=%s phase=%s text=%s",
                guild.id,
                user.id,
                phase,
                text,
            )
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

    listener_started_at = time.monotonic()
    AUTO_LISTEN_SESSIONS[guild.id] = {
        "voice_client": voice_client,
        "guild": guild,
        "text_channel": text_channel,
        "listener_token": listener_token,
        "busy": False,
        "last_text": "",
        "started_at": listener_started_at,
    }

    def after_auto_listen(exc: Exception | None) -> None:
        try:
            sink.cleanup()
        except Exception:
            logger.exception("Failed cleaning up auto-listen sink guild=%s", guild.id)
        if exc:
            logger.exception("Auto-listen callback reported an error guild=%s", guild.id, exc_info=exc)
        current = AUTO_LISTEN_SESSIONS.get(guild.id)
        if (
            current
            and current.get("voice_client") is voice_client
            and current.get("listener_token") == listener_token
            and guild.id not in ACTIVE_RECORDINGS
        ):
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
    pending_exit = PENDING_VOICE_EXIT_TASKS.pop(member.guild.id, None)
    if pending_exit is not None:
        pending_exit.cancel()
    if after.channel is None:
        PENDING_VOICE_EXIT_TASKS[member.guild.id] = asyncio.create_task(
            finalize_voice_exit_if_still_disconnected(
                member.guild,
                before.channel.id if before.channel else None,
            )
        )


@client.event
async def on_message(message: discord.Message) -> None:
    if message.author == client.user:
        return

    if not isinstance(message.channel, discord.abc.Messageable):
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

    if await maybe_reject_message_during_tts(message, command):
        return

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

    if command == "!20q":
        subcommand = parts[1].strip().lower() if len(parts) == 2 else ""
        if subcommand in {"stop", "end", "quit", "end game"}:
            if not await stop_twenty_questions(message, message="20 Questions ended."):
                await message.channel.send("There isn't an active 20 Questions game right now.")
            return
        if subcommand == "status":
            game = ACTIVE_TWENTY_QUESTIONS.get(message.guild.id) if message.guild else None
            if not game or game.get("channel_id") != message.channel.id:
                await message.channel.send("There isn't an active 20 Questions game in this channel.")
                return
            turns_used = game.get("turns_used", 0)
            awaiting = game.get("awaiting", "answer")
            await message.channel.send(
                f"20 Questions is active. Turn {turns_used}/{TWENTY_QUESTIONS_MAX_TURNS}. Waiting for {'a guess result' if awaiting == 'guess_result' else 'an answer'}."
            )
            return
        await start_twenty_questions(message)
        return

    if command in {"!endgame", "!stopgame"}:
        if not await stop_twenty_questions(message, message="20 Questions ended."):
            await message.channel.send("There isn't an active 20 Questions game right now.")
        return

    if command == "!talk" and len(parts) == 1:
        await enable_hands_free_mode(message)
        return

    if command in {"!endtalk", "!stoptalk", "!talkoff"}:
        disabled = False
        if message.guild and stop_auto_listen(message.guild.id, disable=True):
            disabled = True

        voice_client = get_current_voice_client(message.guild)
        if voice_client and voice_client.is_connected():
            if voice_client.is_playing():
                stop_idle_keepalive(voice_client)
                if hasattr(voice_client, "stop_playing"):
                    voice_client.stop_playing()
                else:
                    voice_client.stop()
            start_idle_keepalive(voice_client)

        if disabled:
            await message.channel.send("Hands-free mode is off.")
        else:
            await message.channel.send("Hands-free mode wasn't active.")
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
        if await stop_twenty_questions(message, message="20 Questions ended."):
            stopped_anything = True
        if message.guild:
            LAST_TTS_BUSY_NOTICE_AT.pop(message.guild.id, None)
            was_tts_playing = message.guild.id in ACTIVE_TTS_PLAYBACKS
            ACTIVE_TTS_PLAYBACKS.pop(message.guild.id, None)

        if message.guild and stop_active_recording(message.guild.id):
            stopped_anything = True

        if message.guild:
            if was_tts_playing:
                stop_auto_listen(message.guild.id, disable=False)
            else:
                was_hands_free = message.guild.id in AUTO_LISTEN_ENABLED_GUILDS
                if stop_auto_listen(message.guild.id, disable=True):
                    stopped_anything = True
                if was_hands_free:
                    await message.channel.send("Hands-free mode is off.")

        voice_client = get_current_voice_client(message.guild)
        if voice_client and voice_client.is_playing():
            logger.info(
                "Stop requested for active playback in guild=%s state=%s",
                message.guild.id,
                voice_state_snapshot(voice_client),
            )
            stop_idle_keepalive(voice_client)
            if voice_client.is_playing():
                if hasattr(voice_client, "stop_playing"):
                    voice_client.stop_playing()
                else:
                    voice_client.stop()
            if (
                message.guild
                and message.guild.id not in AUTO_LISTEN_ENABLED_GUILDS
                and voice_client.is_connected()
            ):
                start_idle_keepalive(voice_client)
            stopped_anything = True

        if not stopped_anything:
            await send_error(message.channel, "There is no active talk session, recording, or playback to stop.")
        return

    if command == "!say":
        question = content[4:].strip()
        if not question:
            await message.channel.send("Usage: `!say your message here`")
            return

        if is_twenty_questions_start_request(question):
            await start_twenty_questions(message)
            return

        if is_twenty_questions_end_request(question):
            if not await stop_twenty_questions(message, message="20 Questions ended."):
                await message.channel.send("There isn't an active 20 Questions game right now.")
            return

        if await maybe_handle_twenty_questions_reply(message, question):
            return

        if await maybe_handle_fun_trigger(message, question):
            return

        await respond_with_ollama(
            message,
            question,
            speak_reply=True,
            log_source="text",
        )
        return

    if command.startswith("!"):
        return

    if is_twenty_questions_start_request(content):
        await start_twenty_questions(message)
        return

    if is_twenty_questions_end_request(content):
        if not await stop_twenty_questions(message, message="20 Questions ended."):
            await message.channel.send("There isn't an active 20 Questions game right now.")
        return

    if await maybe_handle_twenty_questions_reply(message, content):
        return

    if await maybe_handle_fun_trigger(message, content):
        return


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
