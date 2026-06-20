# -*- coding: utf-8 -*-
import os
import io
import re
import sys
import logging
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler,
    MessageHandler, filters, ContextTypes
)

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

ALLOWED_USERS_FILE = 'allowed_users.txt'


def load_allowed_users():
    """Reads allowed_users.txt and returns a set of allowed Telegram user IDs.
    One ID per line. Lines starting with # or empty lines are ignored.
    If the file is missing, the bot stays open to everyone (logs a warning)."""
    if not os.path.exists(ALLOWED_USERS_FILE):
        logger.warning(
            f"{ALLOWED_USERS_FILE} not found - whitelist is DISABLED, "
            "bot is open to everyone."
        )
        return None

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

    logger.info(f"Loaded {len(allowed)} allowed user ID(s) from {ALLOWED_USERS_FILE}")
    return allowed


ALLOWED_USERS = load_allowed_users()


def restricted(handler):
    """Decorator: blocks the handler for any user_id not in ALLOWED_USERS.
    If ALLOWED_USERS is None (file missing), the whitelist is skipped entirely."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if ALLOWED_USERS is None:
            return await handler(update, context)

        user_id = update.effective_user.id

        if user_id not in ALLOWED_USERS:
            logger.warning(f"Blocked access attempt from user_id={user_id}")
            await update.message.reply_text(
                "Sorry, this bot is private and only available to approved users."
            )
            return

        return await handler(update, context)

    return wrapper


# Available models
AVAILABLE_MODELS = {
    'flash25': 'gemini-2.5-flash',
    'flash25lite': 'gemini-2.5-flash-lite',
    'flash20': 'gemini-2.0-flash',
    'flash20lite': 'gemini-2.0-flash-lite',
}

# Default model
DEFAULT_MODEL = 'flash25'

user_histories = {}
user_models = {}

STYLES = {
    '--photo':   'photography, realistic, 8k, natural lighting',
    '--anime':   'anime style, vibrant colors, studio quality',
    '--art':     'digital art, artstation, concept art, detailed',
    '--dark':    'dark fantasy, moody, dramatic lighting',
    '--minimal': 'minimalist, clean, simple, white background',
}


def get_user_model(user_id):
    model_key = user_models.get(user_id, DEFAULT_MODEL)
    model_name = AVAILABLE_MODELS[model_key]
    return genai.GenerativeModel(model_name)


# Characters that MarkdownV2 requires to be escaped with a backslash.
# See: https://core.telegram.org/bots/api#markdownv2-style
_MDV2_SPECIAL_CHARS = r'_[]()~`>#+-=|{}.!'


def _escape_plain(text: str) -> str:
    """Escapes all MarkdownV2 special characters in plain (non-formatted) text."""
    return re.sub(f'([{re.escape(_MDV2_SPECIAL_CHARS)}])', r'\\\1', text)


def gemini_markdown_to_telegram(text: str) -> str:
    """Converts Gemini's Markdown (**bold**, *italic*, `code`, ```code blocks```)
    into Telegram's MarkdownV2 format.

    Uses a small recursive scanner rather than a single regex pass, because
    Gemini sometimes nests markers (e.g. "*italic with **bold** inside*"),
    which a flat regex can't parse correctly and ends up producing unbalanced
    output that Telegram then refuses to render.
    """
    # Code blocks/inline code are stashed first so their contents are never
    # touched by the bold/italic scanner or the plain-text escaper below.
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
                buf.append('*' + inner + '*')  # MarkdownV2 bold = single *
                continue
            if text[i] == '*':
                if stop_single:
                    return ''.join(buf)
                i += 1
                inner = parse(stop_double=False, stop_single=True)
                if i < n and text[i] == '*':
                    i += 1
                buf.append('_' + inner + '_')  # MarkdownV2 italic = single _
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


async def send_markdown_safe(message, text: str):
    """Tries to send text formatted as MarkdownV2. If parsing fails for any
    reason, falls back to plain text so the bot never crashes on a reply."""
    try:
        converted = gemini_markdown_to_telegram(text)
        await message.reply_text(converted, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        logger.warning(f"MarkdownV2 send failed, falling back to plain text: {e}")
        await message.reply_text(text)


@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I'm an AI assistant powered by Gemini.\n\n"
        "Just write something - I'll reply\n"
        "/image [description] - generate an image\n"
        "Styles: --photo --anime --art --dark --minimal\n"
        "/model - show current model\n"
        "/models - list available models\n"
        "/setmodel [name] - switch model\n"
        "/clear - clear history\n"
        "/help - all commands"
    )


@restricted
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n\n"
        "Any text - chat with Gemini AI\n\n"
        "Models:\n"
        "/models - list available models\n"
        "/model - current model\n"
        "/setmodel flash25\n"
        "/setmodel flash25lite\n"
        "/setmodel flash20\n"
        "/setmodel flash20lite\n\n"
        "Image generation:\n"
        "/image cat on a roof --anime\n"
        "/image mountain sunset --photo\n"
        "/image abstract shapes --art\n\n"
        "/clear - clear history\n"
        "/help - this help message"
    )


@restricted
async def models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current_model = user_models.get(
        update.message.from_user.id,
        DEFAULT_MODEL
    )

    text = (
        "Available models:\n\n"
        f"• flash25 → {AVAILABLE_MODELS['flash25']}\n"
        f"• flash25lite → {AVAILABLE_MODELS['flash25lite']}\n"
        f"• flash20 → {AVAILABLE_MODELS['flash20']}\n"
        f"• flash20lite → {AVAILABLE_MODELS['flash20lite']}\n\n"
        f"Current model: {current_model}"
    )

    await update.message.reply_text(text)


@restricted
async def current_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    model_key = user_models.get(user_id, DEFAULT_MODEL)

    await update.message.reply_text(
        f"Current model:\n"
        f"{AVAILABLE_MODELS[model_key]}"
    )


@restricted
async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id

    if not context.args:
        await update.message.reply_text(
            "Specify a model.\n\n"
            "Example:\n"
            "/setmodel flash25\n\n"
            "Available models:\n"
            + '\n'.join(AVAILABLE_MODELS.keys())
        )
        return

    selected = context.args[0].lower()

    if selected not in AVAILABLE_MODELS:
        await update.message.reply_text(
            "Unknown model.\n\n"
            "Available models:\n"
            + '\n'.join(AVAILABLE_MODELS.keys())
        )
        return

    user_models[user_id] = selected

    await update.message.reply_text(
        f"Model switched to:\n"
        f"{AVAILABLE_MODELS[selected]}"
    )


@restricted
async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_histories[user_id] = []

    await update.message.reply_text("History cleared!")


@restricted
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_message = update.message.text

    if user_id not in user_histories:
        user_histories[user_id] = []

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )

    try:
        gemini = get_user_model(user_id)

        session = gemini.start_chat(
            history=user_histories[user_id]
        )

        response = session.send_message(user_message)

        user_histories[user_id] = list(session.history)[-20:]

        current = user_models.get(user_id, DEFAULT_MODEL)

        full_response = (
            f"{response.text}\n\n"
            f"🤖 Model: {AVAILABLE_MODELS[current]}"
        )

        MAX_LENGTH = 4000

        for i in range(0, len(full_response), MAX_LENGTH):
            chunk = full_response[i:i + MAX_LENGTH]
            await send_markdown_safe(update.message, chunk)

    except Exception as e:
        logger.error(f"Chat error: {e}")

        await update.message.reply_text(
            "Error reaching Gemini. Please try again."
        )


@restricted
async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Specify a description.\nExample: /image sunset over mountains"
        )
        return

    full_text = ' '.join(context.args)
    style_suffix = ''

    for key, value in STYLES.items():
        if key in full_text:
            full_text = full_text.replace(key, '').strip()
            style_suffix = value
            break

    prompt = f"{full_text}, {style_suffix}" if style_suffix else full_text

    msg = await update.message.reply_text(
        "Generating image, please wait ~20 seconds..."
    )

    try:
        import urllib.parse
        import random
        encoded = urllib.parse.quote(prompt, safe='')
        seed = random.randint(1, 99999)
        pk_key = os.getenv('POLLINATIONS_KEY', '')
        url = f"https://gen.pollinations.ai/image/{encoded}?seed={seed}&nologo=true&key={pk_key}"

        logger.info(f"Pollinations request: {url}")

        response = requests.get(url, timeout=120)

        if response.status_code == 200:
            await update.message.reply_photo(
                photo=io.BytesIO(response.content),
                caption=full_text
            )
            await msg.delete()
        else:
            await msg.edit_text(
                f"Generation error: {response.status_code}. Please try again."
            )

    except requests.Timeout:
        await msg.edit_text("Timed out. Please try again.")
    except Exception as e:
        logger.error(f"Image error: {e}")
        await msg.edit_text(f"Error: {str(e)}")


def main():
    token = os.getenv('TELEGRAM_TOKEN')

    if not token:
        raise ValueError(
            "TELEGRAM_TOKEN not found in .env file"
        )

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear))

    # Models
    app.add_handler(CommandHandler("models", models))
    app.add_handler(CommandHandler("model", current_model))
    app.add_handler(CommandHandler("setmodel", set_model))

    # Images
    app.add_handler(CommandHandler("image", image))

    # Chat
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        chat
    ))

    logger.info("Bot started. Waiting for messages...")

    app.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == '__main__':
    main()
