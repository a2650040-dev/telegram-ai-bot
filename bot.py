# -*- coding: utf-8 -*-
import os
import io
import re
import sys
import json
import wave
import asyncio
import logging
from collections import defaultdict, Counter
from time import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)

from memory import (
    init_db, save_message, get_recent_history,
    get_profile, maybe_refresh_summary, clear_history,
    get_user_settings, set_user_model, set_user_voice_mode,
    delete_user_data, delete_last_messages, set_user_search_mode,
    set_user_role,
)
from roles import ROLES, DEFAULT_ROLE, find_role_key

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

ALLOWED_USERS_FILE = 'allowed_users.txt'

# ── Public / trusted access ──────────────────────────────────────────────

def load_allowed_users():
    """Reads allowed_users.txt -> set of Telegram user IDs treated as
    'trusted'. One ID per line, '#' comments allowed. Missing file -> empty set."""
    if not os.path.exists(ALLOWED_USERS_FILE):
        logger.warning(f"{ALLOWED_USERS_FILE} not found - no trusted users configured.")
        return set()

    allowed = set()
    with open(ALLOWED_USERS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                allowed.add(int(line))
            except ValueError:
                logger.warning(f"Skipping invalid line in {ALLOWED_USERS_FILE}: {line!r}")

    logger.info(f"Loaded {len(allowed)} trusted user ID(s) from {ALLOWED_USERS_FILE}")
    return allowed


TRUSTED_USERS = load_allowed_users()
PUBLIC_MODE = os.getenv('PUBLIC_MODE', 'false').lower() == 'true'
ADMIN_USER_ID = int(os.getenv('ADMIN_USER_ID', '0') or 0)

logger.info(f"PUBLIC_MODE={PUBLIC_MODE}")


def is_allowed(user_id: int) -> bool:
    """Can this user talk to the bot at all?"""
    if PUBLIC_MODE:
        return True
    return user_id in TRUSTED_USERS


def is_trusted(user_id: int) -> bool:
    """Does this user get access to the higher-tier models?"""
    return user_id in TRUSTED_USERS


def restricted(handler):
    """Blocks the handler for anyone not allowed at all (see is_allowed)."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        if not is_allowed(user_id):
            logger.warning(f"Blocked access attempt from user_id={user_id}")
            await update.effective_message.reply_text(
                "Sorry, this bot is private and only available to approved users."
            )
            return
        await _ensure_user_settings_hydrated(user_id)
        return await handler(update, context)
    return wrapper


# ── Models ────────────────────────────────────────────────────────────────
# Free-tier lineup as of October 2026 (ai.google.dev/gemini-api/docs/pricing).
# Gemini 2.x models are legacy: Google only keeps them for existing users.
# Flash-Lite has a much bigger free daily quota than Flash, so it is the default.

PUBLIC_MODELS = {
    'flash35lite': ('gemini-3.5-flash-lite', 'Flash-Lite 3.5'),
    'flash31lite': ('gemini-3.1-flash-lite', 'Flash-Lite 3.1'),
}

TRUSTED_MODELS = {
    'flash38': ('gemini-3.8-flash', 'Flash 3.8'),
    'flash36': ('gemini-3.6-flash', 'Flash 3.6'),
}

ALL_MODELS = {**PUBLIC_MODELS, **TRUSTED_MODELS}
DEFAULT_MODEL = 'flash35lite'
TTS_MODEL = 'gemini-3.8-flash-lite-tts'
TTS_VOICE = 'Kore'

# Used for the "current date and time" line the model gets with every request.
# Change it in Railway -> Variables (BOT_TIMEZONE), e.g. Asia/Yakutsk, Europe/Berlin.
BOT_TIMEZONE = os.getenv('BOT_TIMEZONE', 'Asia/Yakutsk')
# Kyiv was renamed from Kiev in the time zone database; try both spellings.
_TZ_ALIASES = {'Europe/Kiev': 'Europe/Kyiv', 'Europe/Kyiv': 'Europe/Kiev'}


def _resolve_timezone():
    """Returns (tzinfo, name). Falls back to UTC (with a warning) if the name
    isn't known on this machine - add 'tzdata' to requirements.txt if so."""
    for name in (BOT_TIMEZONE, _TZ_ALIASES.get(BOT_TIMEZONE)):
        if not name:
            continue
        try:
            return ZoneInfo(name), name
        except Exception:
            continue
    logger.warning(f"Unknown time zone {BOT_TIMEZONE!r} - using UTC. Check BOT_TIMEZONE / add tzdata to requirements.txt")
    return timezone.utc, 'UTC'


def current_datetime_note() -> str:
    """One line telling the model today's date/time - Gemini has no clock and
    otherwise assumes it is still the year its training data ended."""
    tz, tz_name = _resolve_timezone()
    now = datetime.now(tz)
    return f"Current date and time: {now.strftime('%A, %d %B %Y, %H:%M')} ({tz_name})."

# Conversation history is kept per (user_id, role_key): every role has its own.
user_histories = {}
user_models = {}
user_roles = {}               # user_id -> role key the user is talking to (default role if missing)
user_voice_mode = {}          # user_id -> 'text' (default) or 'voice'
user_search = {}              # user_id -> True when Google Search is switched on
user_pending_image_style = {}  # user_id -> style suffix waiting for a description

_settings_hydrated_users = set()  # user_ids we've already tried to load settings for this run


async def _ensure_user_settings_hydrated(user_id):
    """Loads model_key/voice_mode/search_mode/role_key from Postgres into the
    in-memory dicts, once per user per process. Everything after this keeps
    reading/writing the plain dicts as before - this just stops them
    resetting on deploy."""
    if user_id in _settings_hydrated_users:
        return
    _settings_hydrated_users.add(user_id)
    try:
        settings = await get_user_settings(user_id)
        if settings:
            if settings['model_key']:
                user_models[user_id] = settings['model_key']
            if settings['voice_mode']:
                user_voice_mode[user_id] = settings['voice_mode']
            if settings['search_mode']:
                user_search[user_id] = (settings['search_mode'] == 'on')
            if settings['role_key']:
                user_roles[user_id] = settings['role_key']
    except Exception:
        logger.exception(f"Could not load saved settings from DB for user_id={user_id}")


def available_models_for(user_id):
    if is_trusted(user_id):
        return ALL_MODELS
    return PUBLIC_MODELS


def get_user_model_key(user_id):
    key = user_models.get(user_id, DEFAULT_MODEL)
    # Guard against a trusted-only model lingering for a user who lost trust
    if key not in available_models_for(user_id):
        key = DEFAULT_MODEL
    return key


def get_user_model_name(user_id):
    return ALL_MODELS[get_user_model_key(user_id)][0]


def get_user_role_key(user_id):
    """The role the user is talking to. Falls back to the default role if the
    saved role no longer exists (e.g. its prompt file was removed)."""
    key = user_roles.get(user_id, DEFAULT_ROLE)
    return key if key in ROLES else DEFAULT_ROLE


def role_label(role_key):
    return ROLES.get(role_key, ROLES[DEFAULT_ROLE])['label']


STYLES = {
    'photo':   ('📷 Photo', 'photography, realistic, 8k, natural lighting'),
    'anime':   ('🎨 Anime', 'anime style, vibrant colors, studio quality'),
    'art':     ('🖌 Art', 'digital art, artstation, concept art, detailed'),
    'dark':    ('🌑 Dark', 'dark fantasy, moody, dramatic lighting'),
    'minimal': ('⚪ Minimal', 'minimalist, clean, simple, white background'),
    'none':    ('⭐ Custom', ''),
}
# Kept for users who still type the old --flags with /image
LEGACY_STYLE_FLAGS = {
    '--photo': 'photo', '--anime': 'anime', '--art': 'art',
    '--dark': 'dark', '--minimal': 'minimal',
}


# ── Rate limiting (in-memory; resets on restart) ────────────────────────

CHAT_HOUR_LIMIT = 20
CHAT_DAY_LIMIT = 70
IMAGE_DAY_LIMIT = 10
GLOBAL_DAY_LIMIT = 400

HOUR = 3600
DAY = 86400

chat_timestamps = defaultdict(list)     # user_id -> [timestamps]
image_timestamps = defaultdict(list)    # user_id -> [timestamps]
global_timestamps = []                  # all requests bot-wide

STATS = {
    'chat_total': 0,
    'image_total': 0,
    'voice_total': 0,
    'by_user': Counter(),
    'by_model': Counter(),
}


def _prune(timestamps, window):
    cutoff = time() - window
    while timestamps and timestamps[0] < cutoff:
        timestamps.pop(0)


ADMIN_IDS = {8634926425}  # <-- сюда свой Telegram user_id

def check_chat_rate_limit(user_id) -> str | None:
    """Returns an error message if the user is rate-limited, else None."""
    if user_id in ADMIN_IDS:
        return None
    now = time()
    ts = chat_timestamps[user_id]
    _prune(ts, DAY)
    if sum(1 for t in ts if t > now - HOUR) >= CHAT_HOUR_LIMIT:
        return "Too many messages in the last hour. Please try again a bit later 🙂"
    if len(ts) >= CHAT_DAY_LIMIT:
        return "Daily message limit reached. Come back tomorrow!"
    return None


def check_image_rate_limit(user_id) -> str | None:
    ts = image_timestamps[user_id]
    _prune(ts, DAY)
    if len(ts) >= IMAGE_DAY_LIMIT:
        return "Daily image generation limit reached. Try again tomorrow!"
    return None


def check_global_rate_limit() -> str | None:
    _prune(global_timestamps, DAY)
    if len(global_timestamps) >= GLOBAL_DAY_LIMIT:
        return "The bot has hit its overall daily limit for today. Please try again tomorrow!"
    return None


def record_chat_request(user_id, model_key):
    now = time()
    chat_timestamps[user_id].append(now)
    global_timestamps.append(now)
    STATS['chat_total'] += 1
    STATS['by_user'][user_id] += 1
    STATS['by_model'][model_key] += 1


def record_image_request(user_id):
    now = time()
    image_timestamps[user_id].append(now)
    global_timestamps.append(now)
    STATS['image_total'] += 1
    STATS['by_user'][user_id] += 1


def record_voice_request():
    STATS['voice_total'] += 1


# ── Markdown -> Telegram MarkdownV2 (unchanged from before) ─────────────

_MDV2_SPECIAL_CHARS = r'_[]()~`>#+-=|{}.!'


def _escape_plain(text: str) -> str:
    return re.sub(f'([{re.escape(_MDV2_SPECIAL_CHARS)}])', r'\\\1', text)


def gemini_markdown_to_telegram(text: str) -> str:
    placeholders = []

    def stash_code(match):
        placeholders.append(match.group(0))
        return f'\x00{len(placeholders) - 1}\x00'

    text = re.sub(r'```.*?```', stash_code, text, flags=re.DOTALL)
    text = re.sub(r'`[^`\x00]*?`', stash_code, text)

    i = 0
    n = len(text)

    def parse(stop_double, stop_single):
        nonlocal i
        buf = []
        while i < n:
            if text[i:i + 2] == '**':
                if stop_double:
                    return ''.join(buf)
                i += 2
                inner = parse(stop_double=True, stop_single=False)
                if text[i:i + 2] == '**':
                    i += 2
                buf.append('*' + inner + '*')
                continue
            if text[i] == '*':
                if stop_single:
                    return ''.join(buf)
                i += 1
                inner = parse(stop_double=False, stop_single=True)
                if i < n and text[i] == '*':
                    i += 1
                buf.append('_' + inner + '_')
                continue
            if text[i] == '\x00':
                end = text.index('\x00', i + 1)
                idx = int(text[i + 1:end])
                buf.append(placeholders[idx])
                i = end + 1
                continue
            buf.append(_escape_plain(text[i]))
            i += 1
        return ''.join(buf)

    return parse(stop_double=False, stop_single=False)


def tts_button_markup():
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔊 Read aloud", callback_data="tts:read")]])


async def send_markdown_safe(message, text: str, reply_markup=None):
    """Sends text as MarkdownV2, falling back to plain text if parsing fails."""
    try:
        converted = gemini_markdown_to_telegram(text)
        await message.reply_text(converted, parse_mode=ParseMode.MARKDOWN_V2, reply_markup=reply_markup)
    except Exception as e:
        logger.warning(f"MarkdownV2 send failed, falling back to plain text: {e}")
        await message.reply_text(text, reply_markup=reply_markup)


# ── Text-to-speech / speech understanding helpers ───────────────────────

def synthesize_speech(text: str) -> tuple[bytes, int]:
    """Calls Gemini TTS and returns (OGG/Opus-encoded audio bytes, duration in seconds),
    ready to be sent as a real Telegram voice message (waveform bubble) via reply_voice."""
    response = gemini_client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
                )
            ),
        ),
    )
    pcm_data = response.candidates[0].content.parts[0].inline_data.data

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm_data)
    wav_buffer.seek(0)

    from pydub import AudioSegment
    audio = AudioSegment.from_wav(wav_buffer)
    duration_seconds = int(len(audio) / 1000)
    ogg_buffer = io.BytesIO()
    audio.export(ogg_buffer, format="ogg", codec="libopus")
    return ogg_buffer.getvalue(), duration_seconds


def strip_trailer(text: str) -> str:
    """Removes the footer (sources, search note, '🤖 Model: ...') before sending text to TTS."""
    return re.sub(r'\n*(?:🔎 Sources:|ℹ️? Google Search|🤖 Model:).*$', '', text, flags=re.DOTALL).strip()


def model_footer_line(model_key, role_key) -> str:
    """'🤖 Model: gemini-... · 🧭 Орион' (the role part only for non-default roles).
    strip_trailer() removes the whole line before text goes to TTS."""
    line = f"🤖 Model: {ALL_MODELS[model_key][0]}"
    if role_key != DEFAULT_ROLE:
        line += f" · {role_label(role_key)}"
    return line


# ── Inline keyboards ─────────────────────────────────────────────────────

def main_menu_markup():
    keyboard = [
        [InlineKeyboardButton("ℹ️ Help", callback_data="nav:help")],
        [InlineKeyboardButton("🎭 Roles", callback_data="nav:roles")],
        [
            InlineKeyboardButton("🤖 Models", callback_data="nav:models"),
            InlineKeyboardButton("🎙 Voice", callback_data="nav:voice"),
        ],
        [
            InlineKeyboardButton("🖼 Images", callback_data="nav:image"),
            InlineKeyboardButton("🧹 Clear history", callback_data="nav:clear"),
        ],
        [InlineKeyboardButton("🔎 Google Search", callback_data="nav:search")],
        [InlineKeyboardButton("↩️ Undo last exchange", callback_data="nav:undo")],
    ]
    return InlineKeyboardMarkup(keyboard)


def back_button():
    return [InlineKeyboardButton("⬅️ Back", callback_data="nav:main")]


def models_menu_markup(user_id):
    current = get_user_model_key(user_id)
    rows = []
    for key, (_, label) in available_models_for(user_id).items():
        prefix = "✅ " if key == current else ""
        rows.append([InlineKeyboardButton(f"{prefix}{label}", callback_data=f"model:{key}")])
    rows.append(back_button())
    return InlineKeyboardMarkup(rows)


def roles_menu_text(user_id, notice='') -> str:
    current = get_user_role_key(user_id)
    lines = []
    if notice:
        lines += [notice, ""]
    lines.append("🎭 Roles - every role keeps its own conversation history; your profile is shared.")
    lines.append("")
    for key, role in ROLES.items():
        marker = " (current)" if key == current else ""
        lines.append(f"{role['label']} - {role['description']}{marker}")
    return "\n".join(lines)


def roles_menu_markup(user_id):
    current = get_user_role_key(user_id)
    rows = []
    for key, role in ROLES.items():
        prefix = "✅ " if key == current else ""
        rows.append([InlineKeyboardButton(f"{prefix}{role['label']}", callback_data=f"role:{key}")])
    rows.append(back_button())
    return InlineKeyboardMarkup(rows)


def voice_menu_markup(user_id):
    mode = user_voice_mode.get(user_id, 'text')
    rows = [
        [InlineKeyboardButton(
            ("✅ " if mode == 'text' else "") + "🔤 Voice messages → text reply",
            callback_data="voice:text")],
        [InlineKeyboardButton(
            ("✅ " if mode == 'voice' else "") + "🎙 Voice messages → voice reply",
            callback_data="voice:voice")],
        back_button(),
    ]
    return InlineKeyboardMarkup(rows)


def search_status_text(user_id) -> str:
    if user_search.get(user_id, False):
        return (
            "🔎 Google Search: ON\n\n"
            "I'll check the web for fresh information when it helps. "
            "Not used for voice messages. If search isn't available for the bot's API key, "
            "I'll say so and answer without it."
        )
    return (
        "🔎 Google Search: OFF\n\n"
        "I answer from my own knowledge, but I'll search the web when you ask me to "
        "(e.g. \"поищи в интернете ...\"). Turn it on to search for every message."
    )


def search_menu_markup(user_id):
    on = user_search.get(user_id, False)
    label = "🔎 Turn OFF" if on else "🔎 Turn ON"
    return InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data="search:toggle")], back_button()])


def image_menu_markup():
    rows = []
    row = []
    for key, (label, _) in STYLES.items():
        row.append(InlineKeyboardButton(label, callback_data=f"image:{key}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append(back_button())
    return InlineKeyboardMarkup(rows)


HELP_TEXT = (
    "🤗 Welcome! I'm an AI assistant powered by Gemini.\n\n"
    "What I can do:\n"
    "• Chat - just type a message\n"
    "• Roles - talk to a character instead of the plain bot (mentor, reading partner, psychologist, English teacher and more); each role remembers its own conversation. Switch with /role or /menu → 🎭 Roles\n"
    "• Voice - send a voice message, I can reply in text or voice (see /menu → 🎙 Voice)\n"
    "• Images - generate pictures with /image or /menu → 🖼 Images\n"
    "• Photos & PDFs - send a photo or a PDF; add a caption to ask a question about it\n"
    "• Web search - just ask (e.g. \"поищи в интернете ...\" / \"search the web for ...\"), or keep Google Search on for every message with /search\n"
    "• Multiple models - switch anytime with /setmodel or /menu → 🤖 Models\n\n"
    "Commands:\n"
    "/menu - main menu with buttons\n"
    "/role [name] - choose a role (no name shows the list)\n"
    "/models - list available models\n"
    "/model - current model\n"
    "/setmodel <name> - switch model by typing\n"
    "/image <description> - generate an image (add --anime --art --dark --minimal --photo)\n"
    "/search - turn Google Search on or off\n"
    "/clear - clear the conversation history of the current role\n"
    "/undo - remove the last exchange (recovery from a bad reply)\n"
    "/privacy - what I remember about you, and how to delete it\n"
    "/welcome - show this message"
)

PRIVACY_TEXT = (
    "What I remember:\n"
    "• Your recent conversation history - kept separately for each role (used so I have context between messages)\n"
    "• A short profile summary I build up over time (name, interests, style - "
    "so I don't start from zero every conversation). It is shared by all roles.\n"
    "• Your chosen model, role and voice-reply setting\n\n"
    "For voice messages, I store a text transcript of what you said, not the audio itself. "
    "For photos and PDFs, I store a short note (e.g. \"[photo] your caption\") and my reply, "
    "not the file itself. Photos and PDFs you send are passed to Google's Gemini to be read.\n\n"
    "This is stored in a database so it survives restarts - it isn't shared with other users, "
    "and only the bot owner has access to the underlying database.\n\n"
    "/clear wipes the conversation history of the current role only (keeps other roles, your profile and settings).\n"
    "/forget_me wipes everything I have about you - history of all roles, profile, and settings, all at once."
)

START_TEXT = (
    "Hi! I'm an AI assistant powered by Gemini.\n\n"
    "Tap /menu to open the button menu, or just type something and I'll reply."
)

BOT_COMMANDS = [
    ("welcome", "🤗 Welcome!"),
    ("menu", "📋 Main menu"),
    ("role", "🎭 Roles"),
    ("model", "🤖 Show current model"),
    ("setmodel", "🤖 Models"),
    ("image", "🎨 Image"),
    ("search", "🔎 Google Search on/off"),
    ("clear", "🧹 Clear history"),
    ("undo", "↩️ Undo last exchange"),
    ("privacy", "🔒 Privacy"),
    ("forget_me", "🗑️ Forget me"),
]


async def post_init(app: Application):
    """Registers the command list so Telegram shows the native Menu button.
    Wrapped in try/except: a bad command list must never crash the whole bot."""
    from telegram import BotCommand
    try:
        await app.bot.set_my_commands([BotCommand(cmd, desc) for cmd, desc in BOT_COMMANDS])
    except Exception:
        logger.exception("Failed to set bot commands (menu button); continuing without it")

    try:
        await init_db()
    except Exception:
        logger.exception("Failed to initialize memory DB - persistent memory will not work this run")


# ── Command handlers ──────────────────────────────────────────────────────

@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(START_TEXT)


@restricted
async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Main menu:", reply_markup=main_menu_markup())


@restricted
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT)


@restricted
async def models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    current = get_user_model_key(user_id)
    lines = ["Available models:\n"]
    for key, (model_id, label) in available_models_for(user_id).items():
        marker = " (current)" if key == current else ""
        lines.append(f"• {key} -> {model_id}{marker}")
    await update.message.reply_text('\n'.join(lines))


@restricted
async def current_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    key = get_user_model_key(user_id)
    await update.message.reply_text(f"Current model:\n{ALL_MODELS[key][0]}")


async def _apply_model_choice(user_id, selected_key) -> str:
    """Shared validation used by both /setmodel and the button menu."""
    if selected_key not in ALL_MODELS:
        return "Unknown model."
    if selected_key in TRUSTED_MODELS and not is_trusted(user_id):
        return "This model is only available to trusted users.\nTry flash35lite - it's free and unrestricted."
    user_models[user_id] = selected_key
    try:
        await set_user_model(user_id, selected_key)
    except Exception:
        logger.exception(f"Failed to persist model choice for user_id={user_id}")
    return f"Model switched to:\n{ALL_MODELS[selected_key][0]}"


@restricted
async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        await update.message.reply_text("Choose a model:", reply_markup=models_menu_markup(user_id))
        return
    result = await _apply_model_choice(user_id, context.args[0].lower())
    await update.message.reply_text(result)


async def _apply_role_choice(user_id, selected_key) -> str:
    """Shared by /role and the button menu. Switching a role never touches any
    history: every role keeps its own, and coming back continues where it stopped."""
    if selected_key not in ROLES:
        return "Unknown role."
    user_roles[user_id] = selected_key
    try:
        await set_user_role(user_id, selected_key)
    except Exception:
        logger.exception(f"Failed to persist role choice for user_id={user_id}")
    role = ROLES[selected_key]
    return f"Role: {role['label']} - {role['description']}"


@restricted
async def role_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/role -> list, /role Орион (or orion, or обычный бот) -> switch."""
    user_id = update.message.from_user.id
    if not context.args:
        await update.message.reply_text(roles_menu_text(user_id), reply_markup=roles_menu_markup(user_id))
        return
    selected_key = find_role_key(' '.join(context.args))
    if selected_key is None:
        await update.message.reply_text(
            roles_menu_text(user_id, notice="Unknown role. Pick one:"),
            reply_markup=roles_menu_markup(user_id),
        )
        return
    await update.message.reply_text(await _apply_role_choice(user_id, selected_key))


@restricted
async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clears the conversation of the CURRENT role only."""
    user_id = update.message.from_user.id
    role_key = get_user_role_key(user_id)
    user_histories[(user_id, role_key)] = []
    try:
        await clear_history(user_id, role_key)
    except Exception:
        logger.exception(f"Failed to clear DB history for user_id={user_id}, role={role_key}")
    await update.message.reply_text(f"History cleared ({role_label(role_key)})!")


async def _undo_last_exchange(user_id) -> None:
    """Removes the last exchange of the current role from the cache and from Postgres."""
    role_key = get_user_role_key(user_id)
    history_key = (user_id, role_key)
    if user_histories.get(history_key):
        user_histories[history_key] = user_histories[history_key][:-2]
    await delete_last_messages(user_id, count=2, role_key=role_key)


@restricted
async def undo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Removes the last exchange (your message + the bot's reply) from
    memory - both the in-process cache and Postgres. Recovery tool for a
    broken/blocked turn, without wiping the whole conversation like /clear."""
    user_id = update.message.from_user.id
    try:
        await _undo_last_exchange(user_id)
    except Exception:
        logger.exception(f"Failed to delete last messages for user_id={user_id}")
        await update.message.reply_text("Something went wrong removing the last message - please try again.")
        return
    await update.message.reply_text("Removed the last exchange from memory.")


@restricted
async def privacy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(PRIVACY_TEXT)


@restricted
async def forget_me(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    # Clear everything in the in-process caches too, not just the DB -
    # otherwise the old model/voice-mode/history would keep answering
    # from RAM until the next restart even though the DB is clean.
    for history_key in [k for k in user_histories if k[0] == user_id]:
        del user_histories[history_key]
    user_models.pop(user_id, None)
    user_roles.pop(user_id, None)
    user_voice_mode.pop(user_id, None)
    user_search.pop(user_id, None)
    _settings_hydrated_users.discard(user_id)
    try:
        await delete_user_data(user_id)
    except Exception:
        logger.exception(f"Failed to delete user data for user_id={user_id}")
        await update.message.reply_text("Something went wrong clearing your data - please try again.")
        return
    await update.message.reply_text(
        "Done - I've deleted your conversation history (all roles), profile, and settings. "
        "Next message will start completely fresh."
    )


@restricted
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id != ADMIN_USER_ID:
        return  # silently ignore for non-admins
    top_models = ", ".join(f"{k}:{v}" for k, v in STATS['by_model'].most_common(5)) or "-"
    text = (
        "📊 Stats since last restart:\n\n"
        f"Chat messages: {STATS['chat_total']}\n"
        f"Images generated: {STATS['image_total']}\n"
        f"Voice messages handled: {STATS['voice_total']}\n"
        f"Unique users: {len(STATS['by_user'])}\n"
        f"Top models: {top_models}"
    )
    await update.message.reply_text(text)


# If a message asks for a web search in plain words, search is used for that one
# message even when the /search switch is off.
SEARCH_TRIGGER = re.compile(
    r"(поищи|погугли|загугли"
    r"|поиск\w* (?:в )?(?:интернет|сети|гугл)"
    r"|(?:найди|посмотри|проверь|узнай) (?:это |там )?(?:в )?(?:интернет|сети|гугл)"
    r"|в интернете (?:найди|поищи|посмотри|проверь|узнай)"
    r"|search (?:the )?(?:web|online|internet)"
    r"|google (?:it|this)"
    r"|look (?:it |this )?up online)",
    re.IGNORECASE,
)


def wants_search(user_id, text) -> bool:
    """True if Google Search should be used for this message: the /search switch
    is on, or the message itself asks for a web search."""
    return user_search.get(user_id, False) or bool(text and SEARCH_TRIGGER.search(text))


async def _set_search_mode(user_id, on: bool):
    user_search[user_id] = on
    try:
        await set_user_search_mode(user_id, 'on' if on else 'off')
    except Exception:
        logger.exception(f"Failed to persist search mode for user_id={user_id}")


@restricted
async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/search -> toggle, /search on, /search off."""
    user_id = update.message.from_user.id
    arg = context.args[0].lower() if context.args else ''
    if arg in ('on', 'off'):
        await _set_search_mode(user_id, arg == 'on')
    elif not arg:
        await _set_search_mode(user_id, not user_search.get(user_id, False))
    await update.message.reply_text(search_status_text(user_id), reply_markup=search_menu_markup(user_id))


# ── Image generation (shared by /image and the style-button flow) ───────

async def generate_and_send_image(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                   description: str, style_suffix: str = ''):
    user_id = update.effective_user.id

    limit_msg = check_global_rate_limit() or check_image_rate_limit(user_id)
    if limit_msg:
        await update.effective_message.reply_text(limit_msg)
        return

    prompt = f"{description}, {style_suffix}" if style_suffix else description
    msg = await update.effective_message.reply_text("Generating image, please wait ~20 seconds...")

    import urllib.parse
    import random
    import asyncio

    encoded = urllib.parse.quote(prompt, safe='')
    pk_key = os.getenv('POLLINATIONS_KEY', '')

    MAX_ATTEMPTS = 2
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            seed = random.randint(1, 99999)
            url = f"https://gen.pollinations.ai/image/{encoded}?seed={seed}&nologo=true&key={pk_key}"
            logger.info(f"Pollinations request (attempt {attempt}): {url}")
            response = requests.get(url, timeout=120)

            if response.status_code == 200:
                await update.effective_message.reply_photo(
                    photo=io.BytesIO(response.content),
                    caption=description
                )
                await msg.delete()
                record_image_request(user_id)
                return

            logger.warning(f"Pollinations returned {response.status_code} on attempt {attempt}")
            if attempt == MAX_ATTEMPTS:
                await msg.edit_text(f"Generation error: {response.status_code}. Please try again.")

        except requests.Timeout:
            logger.warning(f"Pollinations timed out on attempt {attempt}")
            if attempt == MAX_ATTEMPTS:
                await msg.edit_text("Timed out after retrying. The image service may be overloaded right now - please try again in a bit.")
            else:
                await asyncio.sleep(1)
        except Exception:
            logger.exception("Image error")
            await msg.edit_text("Something went wrong generating the image. Please try again.")
            return


@restricted
async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Pick a style, then type what to draw:",
            reply_markup=image_menu_markup()
        )
        return

    full_text = ' '.join(context.args)
    style_suffix = ''
    for flag, style_key in LEGACY_STYLE_FLAGS.items():
        if flag in full_text:
            full_text = full_text.replace(flag, '').strip()
            style_suffix = STYLES[style_key][1]
            break

    await generate_and_send_image(update, context, full_text, style_suffix)


# ── Chat (text) ────────────────────────────────────────────────────────

LANGUAGE_INSTRUCTION = (
    "Always reply in the same language the user's most recent message is "
    "written in (text or transcribed speech). If the message mixes languages, "
    "use whichever language dominates it."
)

# Added only when a role is active: some roles prescribe their own language
# (e.g. the English teacher explains in Russian and practises in English).
ROLE_LANGUAGE_NOTE = (
    "If the role description above says which language to use, follow the role description."
)

AUDIO_INSTRUCTION = (
    "The user's message is a voice recording. Respond only to the words they "
    "actually spoke. If the recording is too short, silent, or too unclear to "
    "make out real words, say so honestly and ask them to resend a clearer or "
    "longer recording - never guess or comment on technical properties like "
    "the file's duration instead of its content."
)

VOICE_JSON_INSTRUCTION = (
    "You must respond with a single JSON object with exactly two string fields: "
    "'transcript' (a faithful transcription of what the user actually said, in "
    "the language they spoke it in, with no translation and no commentary) and "
    "'reply' (your normal conversational reply to them, following all the rules "
    "above). Do not include anything outside the JSON object."
)

VOICE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "transcript": {"type": "string"},
        "reply": {"type": "string"},
    },
    "required": ["transcript", "reply"],
}


def _parse_voice_turn(raw_text: str):
    """Parses the {"transcript": ..., "reply": ...} JSON we asked Gemini for.
    Returns (reply_text, transcript_text) on success, or None if the model
    didn't return valid/complete JSON - the caller is responsible for
    falling back to a plain-text retry rather than showing the user a raw
    JSON fragment."""
    try:
        data = json.loads(raw_text)
        reply = (data.get("reply") or "").strip()
        transcript = (data.get("transcript") or "").strip()
        if reply:
            return reply, (transcript or "[voice message]")
    except (json.JSONDecodeError, AttributeError):
        pass
    return None


async def _hydrate_history(user_id, role_key):
    """Cold start (fresh process after a deploy/restart): rebuild this
    user's Gemini chat history for one role from Postgres instead of
    starting empty."""
    rows = await get_recent_history(user_id, limit=20, role_key=role_key)
    return [
        types.Content(role=row['role'], parts=[types.Part.from_text(text=row['content'])])
        for row in rows
    ]


async def run_chat_turn(user_id, model_key, content, max_attempts=3,
                        media_label=None, use_search=False, role_key=DEFAULT_ROLE):
    """Runs one conversation turn and returns (reply_text, footer).

    content: a string, or a list of genai Parts (voice, or photo/PDF + question).
    media_label: set for photo/PDF turns - the short text note stored in memory
    instead of the file itself (e.g. "[photo] what is this?").
    use_search: turn on Google Search grounding (text and photo/PDF turns only).
    role_key: which role is talking. The role's prompt goes into the system
    instruction, and the turn is read from / saved to that role's own history.
    footer: extra text to show under the reply ("" normally; the list of web
    sources when search was used, or a note when search had to be skipped).

    Retries automatically on Gemini 503 (overloaded) errors before giving up.
    History and a long-term profile live in Postgres, so this survives
    deploys/restarts - the in-memory dict is just a per-process cache.

    For voice input, asks Gemini for {transcript, reply} in the same call:
    'reply' is what gets sent/spoken back to the user, 'transcript' is what
    gets stored in memory (in place of the raw audio) - one API call, no
    extra latency or meaningful token cost beyond the audio Gemini already
    has to process either way. If Gemini fails to return valid JSON (rare),
    we retry once with a plain-text request instead of ever showing the
    user a raw/truncated JSON fragment."""
    if role_key not in ROLES:
        role_key = DEFAULT_ROLE
    history_key = (user_id, role_key)

    if history_key not in user_histories:
        try:
            user_histories[history_key] = await _hydrate_history(user_id, role_key)
        except Exception:
            logger.exception(f"Could not load history from DB for user_id={user_id}, role={role_key}, starting empty")
            user_histories[history_key] = []

    model_name = ALL_MODELS[model_key][0]
    is_media = media_label is not None
    is_audio = isinstance(content, list) and not is_media

    try:
        profile = await get_profile(user_id)
    except Exception:
        logger.exception(f"Could not load profile for user_id={user_id}")
        profile = None
    profile_note = f" What you know about this user: {profile['summary']}" if profile and profile['summary'] else ""

    role_prompt = ROLES[role_key]['prompt']
    role_block = f"{role_prompt}\n\n" if role_prompt else ""
    language_rules = LANGUAGE_INSTRUCTION + (f" {ROLE_LANGUAGE_NOTE}" if role_prompt else "")

    base_system_instruction = (
        role_block
        + language_rules
        + f" {current_datetime_note()}"
        + profile_note
        + (f" {AUDIO_INSTRUCTION}" if is_audio else "")
    )
    json_system_instruction = base_system_instruction + (f" {VOICE_JSON_INSTRUCTION}" if is_audio else "")

    config_kwargs = {"system_instruction": json_system_instruction}
    if is_audio:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = VOICE_RESPONSE_SCHEMA

    # Google Search is not combined with the voice JSON schema.
    search_enabled = bool(use_search) and not is_audio
    search_fell_back = False
    if search_enabled:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            session = gemini_client.chats.create(
                model=model_name,
                history=user_histories[history_key],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            response = session.send_message(content)

            if is_audio:
                parsed = _parse_voice_turn(response.text)
                if parsed is None:
                    # Gemini didn't comply with the JSON schema - rather than
                    # show the user a broken JSON fragment, ask again plainly.
                    logger.warning(f"Voice turn: invalid JSON from Gemini for user_id={user_id}, retrying without schema")
                    plain_session = gemini_client.chats.create(
                        model=model_name,
                        history=user_histories[history_key],
                        config=types.GenerateContentConfig(system_instruction=base_system_instruction),
                    )
                    plain_response = plain_session.send_message(content)
                    reply_text = plain_response.text
                    stored_user_text = "[voice message]"  # no transcript available this time
                else:
                    reply_text, stored_user_text = parsed
            elif is_media:
                reply_text = response.text
                stored_user_text = media_label
            else:
                reply_text = response.text
                stored_user_text = content

            if not reply_text:
                # Gemini returned no text at all - usually a safety/content
                # block. Bail out here, BEFORE touching user_histories or
                # the DB: session.get_history() at this point would contain
                # an empty/broken turn that poisons every future request for
                # this user (Gemini rejects a history with an empty content
                # entry), which is exactly the "Error reaching Gemini" loop
                # this was causing. Nothing gets saved, history stays as it
                # was before this turn, and the user gets a clear message
                # instead of a literal "None".
                logger.warning(f"Gemini returned no text for user_id={user_id} (likely blocked) - not touching history")
                return "Sorry, I can't respond to that - it looks like it was blocked by a content filter. Try rephrasing.", ""

            if is_audio or is_media:
                # Swap the raw audio/photo/PDF-in turn for clean text
                # equivalents, so future turns don't keep re-sending the file
                # bytes as context (expensive) and history stays readable.
                raw_history = session.get_history()
                base_history = raw_history[:-2] if len(raw_history) >= 2 else []
                base_history.append(types.Content(role='user', parts=[types.Part.from_text(text=stored_user_text)]))
                base_history.append(types.Content(role='model', parts=[types.Part.from_text(text=reply_text)]))
                user_histories[history_key] = base_history[-20:]
            else:
                user_histories[history_key] = session.get_history()[-20:]

            try:
                await save_message(user_id, "user", stored_user_text, role_key)
                await save_message(user_id, "model", reply_text, role_key)
                await maybe_refresh_summary(user_id, gemini_client)
            except Exception:
                logger.exception(f"Failed to persist memory for user_id={user_id} (reply still sent normally)")

            footer = ""
            if search_enabled:
                footer = _sources_footer(response)
            elif search_fell_back:
                footer = "\n\nℹ️ Google Search wasn't available for this request, so I answered without it."
            return reply_text, footer
        except genai_errors.ServerError:
            logger.warning(f"Gemini overloaded (attempt {attempt}/{max_attempts})")
            if attempt == max_attempts:
                raise
            await asyncio.sleep(2 * attempt)
        except genai_errors.ClientError as e:
            if search_enabled:
                # Most likely Google Search isn't allowed for this API key / model
                # (e.g. free tier), or the search quota is used up. Don't fail the
                # whole turn: retry once without the search tool.
                logger.warning(f"Gemini rejected the request with Google Search on ({e}); retrying without search")
                config_kwargs.pop("tools", None)
                search_enabled = False
                search_fell_back = True
                attempt -= 1  # the fallback doesn't use up a retry
                continue
            raise


def _sources_footer(response, limit=5) -> str:
    """Builds a short '🔎 Sources: ...' line from Gemini's grounding metadata.
    Returns "" if the answer wasn't based on any search results."""
    try:
        metadata = response.candidates[0].grounding_metadata
        chunks = (metadata.grounding_chunks or []) if metadata else []
    except Exception:
        return ""
    names = []
    for chunk in chunks:
        web = getattr(chunk, 'web', None)
        title = (getattr(web, 'title', None) or getattr(web, 'uri', None) or '').strip()
        if title and title not in names:
            names.append(title)
        if len(names) >= limit:
            break
    return ("\n\n🔎 Sources: " + ", ".join(names)) if names else ""


async def send_reply_chunks(message, full_response: str, max_length: int = 4000):
    """Splits a long reply into Telegram-sized pieces and sends each with the
    'Read aloud' button."""
    for i in range(0, len(full_response), max_length):
        chunk = full_response[i:i + max_length]
        await send_markdown_safe(message, chunk, reply_markup=tts_button_markup())


@restricted
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_message = update.message.text

    # If the user just tapped an image-style button, this text is the image
    # description they were asked for, not a chat message.
    if user_id in user_pending_image_style:
        style_key = user_pending_image_style.pop(user_id)
        style_suffix = STYLES[style_key][1]
        await generate_and_send_image(update, context, user_message, style_suffix)
        return

    limit_msg = check_global_rate_limit() or check_chat_rate_limit(user_id)
    if limit_msg:
        await update.message.reply_text(limit_msg)
        return

    waiting_msg = await update.message.reply_text("💭 Please wait, preparing a response...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        model_key = get_user_model_key(user_id)
        role_key = get_user_role_key(user_id)
        reply_text, footer = await run_chat_turn(
            user_id, model_key, user_message,
            use_search=wants_search(user_id, user_message),
            role_key=role_key,
        )
        record_chat_request(user_id, model_key)

        await waiting_msg.delete()

        full_response = f"{reply_text}{footer}\n\n{model_footer_line(model_key, role_key)}"
        await send_reply_chunks(update.message, full_response)

    except genai_errors.ServerError:
        logger.exception("Chat error - Gemini overloaded")
        await waiting_msg.edit_text("Google's servers are overloaded right now. Please try again in a minute.")
    except Exception:
        logger.exception("Chat error")
        await waiting_msg.edit_text("Error reaching Gemini. Please try again.")


# ── Voice messages ────────────────────────────────────────────────────

@restricted
async def voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id

    limit_msg = check_global_rate_limit() or check_chat_rate_limit(user_id)
    if limit_msg:
        await update.message.reply_text(limit_msg)
        return

    waiting_msg = await update.message.reply_text("🎤 Processing your voice message...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        voice_file = await update.message.voice.get_file()
        voice_bytes = bytes(await voice_file.download_as_bytearray())
        audio_part = types.Part.from_bytes(data=voice_bytes, mime_type='audio/ogg')

        model_key = get_user_model_key(user_id)
        role_key = get_user_role_key(user_id)
        reply_text, footer = await run_chat_turn(user_id, model_key, [audio_part], role_key=role_key)
        record_chat_request(user_id, model_key)
        record_voice_request()

        mode = user_voice_mode.get(user_id, 'text')

        if mode == 'voice':
            await waiting_msg.edit_text("🔊 Generating voice reply...")
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE)
            audio_bytes, duration = synthesize_speech(strip_trailer(reply_text))
            await waiting_msg.delete()
            await update.message.reply_voice(
                voice=io.BytesIO(audio_bytes),
                filename="reply.ogg",
                duration=duration
            )
        else:
            await waiting_msg.delete()
            full_response = f"{reply_text}{footer}\n\n{model_footer_line(model_key, role_key)}"
            await send_reply_chunks(update.message, full_response)

    except genai_errors.ServerError:
        logger.exception("Voice message error - Gemini overloaded")
        await waiting_msg.edit_text("Google's servers are overloaded right now. Please try again in a minute.")
    except Exception:
        logger.exception("Voice message error")
        await waiting_msg.edit_text("Couldn't process the voice message. Please try again.")


# ── Photos and PDFs ───────────────────────────────────────────────────

SUPPORTED_MEDIA_MIME = {
    'application/pdf',
    'image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif',
}
MAX_MEDIA_BYTES = 15 * 1024 * 1024  # stays under Telegram's 20 MB bot download limit

# Used when the user sends a file without a caption. Change the language here if needed.
DEFAULT_IMAGE_PROMPT = "Опиши, что на этом изображении."
DEFAULT_PDF_PROMPT = "Сделай краткое содержание этого документа."


@restricted
async def media_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """A photo, an image sent as a file, or a PDF. The caption (if any) is the
    question; with no caption the bot describes the image / summarizes the PDF."""
    user_id = update.message.from_user.id
    message = update.message

    if message.photo:
        tg_obj = message.photo[-1]  # largest size
        mime = 'image/jpeg'
        kind_label = "[photo]"
    else:
        tg_obj = message.document
        mime = (tg_obj.mime_type or '').lower()
        if mime not in SUPPORTED_MEDIA_MIME:
            await message.reply_text("I can read photos, images (JPG, PNG, WebP, HEIC) and PDF files.")
            return
        kind_label = f"[PDF: {tg_obj.file_name}]" if mime == 'application/pdf' else "[image file]"

    if tg_obj.file_size and tg_obj.file_size > MAX_MEDIA_BYTES:
        await message.reply_text(f"That file is too large (max {MAX_MEDIA_BYTES // (1024 * 1024)} MB).")
        return

    limit_msg = check_global_rate_limit() or check_chat_rate_limit(user_id)
    if limit_msg:
        await message.reply_text(limit_msg)
        return

    is_pdf = mime == 'application/pdf'
    waiting_msg = await message.reply_text("📄 Reading the document..." if is_pdf else "🖼 Looking at the image...")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        tg_file = await tg_obj.get_file()
        file_bytes = bytes(await tg_file.download_as_bytearray())
        media_part = types.Part.from_bytes(data=file_bytes, mime_type=mime)

        caption = (message.caption or '').strip()
        prompt_text = caption or (DEFAULT_PDF_PROMPT if is_pdf else DEFAULT_IMAGE_PROMPT)
        media_label = f"{kind_label} {caption}".strip()

        model_key = get_user_model_key(user_id)
        role_key = get_user_role_key(user_id)
        reply_text, footer = await run_chat_turn(
            user_id, model_key, [media_part, prompt_text],
            media_label=media_label,
            use_search=wants_search(user_id, caption),
            role_key=role_key,
        )
        record_chat_request(user_id, model_key)

        await waiting_msg.delete()
        full_response = f"{reply_text}{footer}\n\n{model_footer_line(model_key, role_key)}"
        await send_reply_chunks(message, full_response)

    except genai_errors.ServerError:
        logger.exception("Media message error - Gemini overloaded")
        await waiting_msg.edit_text("Google's servers are overloaded right now. Please try again in a minute.")
    except Exception:
        logger.exception("Media message error")
        await waiting_msg.edit_text("Couldn't read that file. Please try again.")


# ── Callback query (button) handler ──────────────────────────────────────

async def _edit_menu_message(query, text, reply_markup):
    """edit_message_text, but a tap that changes nothing (e.g. choosing the role
    that is already selected) makes Telegram answer 'message is not modified' -
    that is not an error worth failing the handler for."""
    try:
        await query.edit_message_text(text, reply_markup=reply_markup)
    except Exception as e:
        logger.info(f"Menu message not edited: {e}")


@restricted
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    if data == "tts:read":
        try:
            await query.answer()
        except Exception:
            logger.warning("Couldn't answer callback query (likely expired) - continuing anyway")
        text = strip_trailer(query.message.text or "")
        if not text:
            return
        waiting_msg = await query.message.reply_text("🔊 Generating audio, please wait...")
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE)
            audio_bytes, duration = synthesize_speech(text)
            await waiting_msg.delete()
            await query.message.reply_voice(voice=io.BytesIO(audio_bytes), filename="reply.ogg", duration=duration)
        except Exception:
            logger.exception("TTS button error")
            await waiting_msg.edit_text("Couldn't generate audio for this reply.")
        return

    await query.answer()

    if data == "nav:main":
        await query.edit_message_text("Main menu:", reply_markup=main_menu_markup())

    elif data == "nav:help":
        await query.edit_message_text(HELP_TEXT, reply_markup=InlineKeyboardMarkup([back_button()]))

    elif data == "nav:roles":
        await _edit_menu_message(query, roles_menu_text(user_id), roles_menu_markup(user_id))

    elif data.startswith("role:"):
        selected_key = data.split(":", 1)[1]
        result = await _apply_role_choice(user_id, selected_key)
        await _edit_menu_message(query, roles_menu_text(user_id, notice=result), roles_menu_markup(user_id))

    elif data == "nav:models":
        await query.edit_message_text("Choose a model:", reply_markup=models_menu_markup(user_id))

    elif data.startswith("model:"):
        selected_key = data.split(":", 1)[1]
        result = await _apply_model_choice(user_id, selected_key)
        await query.edit_message_text(result, reply_markup=models_menu_markup(user_id))

    elif data == "nav:voice":
        await query.edit_message_text(
            "How should I reply to voice messages?",
            reply_markup=voice_menu_markup(user_id)
        )

    elif data in ("voice:text", "voice:voice"):
        user_voice_mode[user_id] = 'voice' if data == "voice:voice" else 'text'
        try:
            await set_user_voice_mode(user_id, user_voice_mode[user_id])
        except Exception:
            logger.exception(f"Failed to persist voice mode for user_id={user_id}")
        await query.edit_message_text(
            "How should I reply to voice messages?",
            reply_markup=voice_menu_markup(user_id)
        )

    elif data == "nav:image":
        await query.edit_message_text(
            "Pick a style, then type what to draw:",
            reply_markup=image_menu_markup()
        )

    elif data.startswith("image:"):
        style_key = data.split(":", 1)[1]
        user_pending_image_style[user_id] = style_key
        label = STYLES[style_key][0]
        await query.edit_message_text(f"Style selected: {label}\n\nNow type what you'd like to draw.")

    elif data == "nav:search":
        await query.edit_message_text(search_status_text(user_id), reply_markup=search_menu_markup(user_id))

    elif data == "search:toggle":
        await _set_search_mode(user_id, not user_search.get(user_id, False))
        await query.edit_message_text(search_status_text(user_id), reply_markup=search_menu_markup(user_id))

    elif data == "nav:clear":
        role_key = get_user_role_key(user_id)
        user_histories[(user_id, role_key)] = []
        try:
            await clear_history(user_id, role_key)
        except Exception:
            logger.exception(f"Failed to clear DB history for user_id={user_id}, role={role_key}")
        await query.edit_message_text(f"History cleared ({role_label(role_key)})!", reply_markup=InlineKeyboardMarkup([back_button()]))

    elif data == "nav:undo":
        try:
            await _undo_last_exchange(user_id)
            await query.edit_message_text("Removed the last exchange from memory.", reply_markup=InlineKeyboardMarkup([back_button()]))
        except Exception:
            logger.exception(f"Failed to delete last messages for user_id={user_id}")
            await query.edit_message_text("Something went wrong removing the last message.", reply_markup=InlineKeyboardMarkup([back_button()]))


# ── App wiring ────────────────────────────────────────────────────────

def main():
    token = os.getenv('TELEGRAM_TOKEN')
    if not token:
        raise ValueError("TELEGRAM_TOKEN not found in .env file")

    app = Application.builder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", menu))
    app.add_handler(CommandHandler("welcome", help_command))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("undo", undo))
    app.add_handler(CommandHandler("privacy", privacy_command))
    app.add_handler(CommandHandler("forget_me", forget_me))
    app.add_handler(CommandHandler("stats", stats_command))

    app.add_handler(CommandHandler("models", models))
    app.add_handler(CommandHandler("model", current_model))
    app.add_handler(CommandHandler("setmodel", set_model))
    app.add_handler(CommandHandler("role", role_command))

    app.add_handler(CommandHandler("image", image))
    app.add_handler(CommandHandler("search", search_command))

    app.add_handler(CallbackQueryHandler(button_callback))

    app.add_handler(MessageHandler(
        filters.PHOTO | filters.Document.PDF | filters.Document.IMAGE, media_message))
    app.add_handler(MessageHandler(filters.VOICE, voice_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    logger.info("Bot started. Waiting for messages...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
