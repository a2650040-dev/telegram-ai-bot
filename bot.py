# -*- coding: utf-8 -*-
import os
import io
import re
import sys
import wave
import logging
from collections import defaultdict, Counter
from time import time

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, filters, ContextTypes
)

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

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
        return await handler(update, context)
    return wrapper


# ── Models ────────────────────────────────────────────────────────────────
# Gemini 2.0 Flash / 2.0 Flash-Lite were shut down by Google on June 1, 2026
# and are removed here. Current free-tier lineup as of mid-2026:

PUBLIC_MODELS = {
    'flash25lite': ('gemini-2.5-flash-lite', 'Flash-Lite 2.5'),
    'flash25':     ('gemini-2.5-flash', 'Flash 2.5'),
}

TRUSTED_MODELS = {
    'flash3preview':    ('gemini-3-flash-preview', 'Flash 3 (Preview)'),
    'flash31litepreview': ('gemini-3.1-flash-lite', 'Flash-Lite 3.1 (Preview)'),
    'pro25':            ('gemini-2.5-pro', 'Pro 2.5'),
}

ALL_MODELS = {**PUBLIC_MODELS, **TRUSTED_MODELS}
DEFAULT_MODEL = 'flash25lite'
TTS_MODEL = 'gemini-2.5-flash-preview-tts'
TTS_VOICE = 'Kore'

user_histories = {}
user_models = {}
user_voice_mode = {}          # user_id -> 'text' (default) or 'voice'
user_pending_image_style = {}  # user_id -> style suffix waiting for a description


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


STYLES = {
    'photo':   ('📷 Photo', 'photography, realistic, 8k, natural lighting'),
    'anime':   ('🎨 Anime', 'anime style, vibrant colors, studio quality'),
    'art':     ('🖌 Art', 'digital art, artstation, concept art, detailed'),
    'dark':    ('🌑 Dark', 'dark fantasy, moody, dramatic lighting'),
    'minimal': ('⚪ Minimal', 'minimalist, clean, simple, white background'),
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


def check_chat_rate_limit(user_id) -> str | None:
    """Returns an error message if the user is rate-limited, else None."""
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

def synthesize_speech(text: str) -> bytes:
    """Calls Gemini TTS and returns WAV-encoded audio bytes."""
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

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm_data)
    return buffer.getvalue()


def strip_trailer(text: str) -> str:
    """Removes the '🤖 Model: ...' footer line before sending text to TTS."""
    return re.sub(r'\n*🤖 Model:.*$', '', text).strip()


# ── Inline keyboards ─────────────────────────────────────────────────────

def main_menu_markup():
    keyboard = [
        [InlineKeyboardButton("ℹ️ Help", callback_data="nav:help")],
        [
            InlineKeyboardButton("🤖 Models", callback_data="nav:models"),
            InlineKeyboardButton("🎙 Voice", callback_data="nav:voice"),
        ],
        [
            InlineKeyboardButton("🖼 Images", callback_data="nav:image"),
            InlineKeyboardButton("🧹 Clear history", callback_data="nav:clear"),
        ],
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
    "Commands:\n\n"
    "Just type a message - I'll reply via Gemini AI\n"
    "Send a voice message - I'll reply with text or voice (configurable in /menu → Voice)\n\n"
    "/menu - main menu with buttons\n"
    "/models - list available models\n"
    "/model - current model\n"
    "/setmodel <name> - switch model by typing\n"
    "/image <description> - generate an image (add --anime --art --dark --minimal --photo)\n"
    "/clear - clear conversation history\n"
    "/help - this message"
)

START_TEXT = (
    "Hi! I'm an AI assistant powered by Gemini.\n\n"
    "Tap /menu to open the button menu, or just type something and I'll reply."
)

BOT_COMMANDS = [
    ("start", "Start the bot"),
    ("menu", "Open the main menu"),
    ("help", "Show help"),
    ("models", "List available models"),
    ("model", "Show current model"),
    ("setmodel", "Switch model by name"),
    ("image", "Generate an image"),
    ("clear", "Clear conversation history"),
]


async def post_init(app: Application):
    """Registers the command list so Telegram shows the native Menu button."""
    from telegram import BotCommand
    await app.bot.set_my_commands([BotCommand(cmd, desc) for cmd, desc in BOT_COMMANDS])


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
        return "This model is only available to trusted users.\nTry flash25lite - it's free and unrestricted."
    user_models[user_id] = selected_key
    return f"Model switched to:\n{ALL_MODELS[selected_key][0]}"


@restricted
async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not context.args:
        available = '\n'.join(available_models_for(user_id).keys())
        await update.message.reply_text(f"Specify a model.\n\nAvailable:\n{available}")
        return
    result = await _apply_model_choice(user_id, context.args[0].lower())
    await update.message.reply_text(result)


@restricted
async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_histories[user_id] = []
    await update.message.reply_text("History cleared!")


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
            "Specify a description.\nExample: /image sunset over mountains --anime\n\n"
            "Or open /menu -> 🖼 Images to pick a style with a button."
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


async def run_chat_turn(user_id, model_key, content):
    """content: a string, or a list of genai Parts (e.g. for voice input)."""
    if user_id not in user_histories:
        user_histories[user_id] = []

    model_name = ALL_MODELS[model_key][0]
    session = gemini_client.chats.create(
        model=model_name,
        history=user_histories[user_id],
        config=types.GenerateContentConfig(system_instruction=LANGUAGE_INSTRUCTION),
    )
    response = session.send_message(content)
    user_histories[user_id] = session.get_history()[-20:]
    return response.text


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

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        model_key = get_user_model_key(user_id)
        reply_text = await run_chat_turn(user_id, model_key, user_message)
        record_chat_request(user_id, model_key)

        full_response = f"{reply_text}\n\n🤖 Model: {ALL_MODELS[model_key][0]}"

        MAX_LENGTH = 4000
        for i in range(0, len(full_response), MAX_LENGTH):
            chunk = full_response[i:i + MAX_LENGTH]
            await send_markdown_safe(update.message, chunk, reply_markup=tts_button_markup())

    except Exception:
        logger.exception("Chat error")
        await update.message.reply_text("Error reaching Gemini. Please try again.")


# ── Voice messages ────────────────────────────────────────────────────

@restricted
async def voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id

    limit_msg = check_global_rate_limit() or check_chat_rate_limit(user_id)
    if limit_msg:
        await update.message.reply_text(limit_msg)
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    try:
        voice_file = await update.message.voice.get_file()
        voice_bytes = bytes(await voice_file.download_as_bytearray())
        audio_part = types.Part.from_bytes(data=voice_bytes, mime_type='audio/ogg')

        model_key = get_user_model_key(user_id)
        reply_text = await run_chat_turn(user_id, model_key, [audio_part])
        record_chat_request(user_id, model_key)
        record_voice_request()

        mode = user_voice_mode.get(user_id, 'text')

        if mode == 'voice':
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE)
            audio_bytes = synthesize_speech(strip_trailer(reply_text))
            await update.message.reply_audio(
                audio=io.BytesIO(audio_bytes),
                filename="reply.wav",
                caption=f"🤖 {ALL_MODELS[model_key][0]}"
            )
        else:
            full_response = f"{reply_text}\n\n🤖 Model: {ALL_MODELS[model_key][0]}"
            MAX_LENGTH = 4000
            for i in range(0, len(full_response), MAX_LENGTH):
                chunk = full_response[i:i + MAX_LENGTH]
                await send_markdown_safe(update.message, chunk, reply_markup=tts_button_markup())

    except Exception:
        logger.exception("Voice message error")
        await update.message.reply_text("Couldn't process the voice message. Please try again.")


# ── Callback query (button) handler ──────────────────────────────────────

@restricted
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data

    if data == "tts:read":
        await query.answer()
        text = strip_trailer(query.message.text or "")
        if not text:
            return
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.RECORD_VOICE)
            audio_bytes = synthesize_speech(text)
            await query.message.reply_audio(audio=io.BytesIO(audio_bytes), filename="reply.wav")
        except Exception:
            logger.exception("TTS button error")
            await query.message.reply_text("Couldn't generate audio for this reply.")
        return

    await query.answer()

    if data == "nav:main":
        await query.edit_message_text("Main menu:", reply_markup=main_menu_markup())

    elif data == "nav:help":
        await query.edit_message_text(HELP_TEXT, reply_markup=InlineKeyboardMarkup([back_button()]))

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

    elif data == "nav:clear":
        user_histories[user_id] = []
        await query.edit_message_text("History cleared!", reply_markup=InlineKeyboardMarkup([back_button()]))


# ── App wiring ────────────────────────────────────────────────────────

def main():
    token = os.getenv('TELEGRAM_TOKEN')
    if not token:
        raise ValueError("TELEGRAM_TOKEN not found in .env file")

    app = Application.builder().token(token).post_init(post_init).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", menu))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("stats", stats_command))

    app.add_handler(CommandHandler("models", models))
    app.add_handler(CommandHandler("model", current_model))
    app.add_handler(CommandHandler("setmodel", set_model))

    app.add_handler(CommandHandler("image", image))

    app.add_handler(CallbackQueryHandler(button_callback))

    app.add_handler(MessageHandler(filters.VOICE, voice_message))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    logger.info("Bot started. Waiting for messages...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()