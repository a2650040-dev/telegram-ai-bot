# -*- coding: utf-8 -*-
import os
import io
import logging
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from telegram import Update
from telegram.ext import (
    Application, CommandHandler,
    MessageHandler, filters, ContextTypes
)

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Доступные модели
AVAILABLE_MODELS = {
    'flash25': 'gemini-2.5-flash',
    'flash25lite': 'gemini-2.5-flash-lite',
    'flash20': 'gemini-2.0-flash',
    'flash20lite': 'gemini-2.0-flash-lite',
}

# Модель по умолчанию
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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я AI-ассистент на базе Gemini.\n\n"
        "Просто напиши - отвечу\n"
        "/image [описание] - сгенерирую картинку\n"
        "Стили: --photo --anime --art --dark --minimal\n"
        "/model - показать текущую модель\n"
        "/models - список моделей\n"
        "/setmodel [название] - сменить модель\n"
        "/clear - очистить историю\n"
        "/help - все команды"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Команды:\n\n"
        "Любой текст - диалог с Gemini AI\n\n"
        "Модели:\n"
        "/models - список моделей\n"
        "/model - текущая модель\n"
        "/setmodel flash25\n"
        "/setmodel flash25lite\n"
        "/setmodel flash20\n"
        "/setmodel flash20lite\n\n"
        "Генерация изображений:\n"
        "/image кот на крыше --anime\n"
        "/image горный закат --photo\n"
        "/image абстракция --art\n\n"
        "/clear - очистить историю\n"
        "/help - эта справка"
    )


async def models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current_model = user_models.get(
        update.message.from_user.id,
        DEFAULT_MODEL
    )

    text = (
        "Доступные модели:\n\n"
        f"• flash25 → {AVAILABLE_MODELS['flash25']}\n"
        f"• flash25lite → {AVAILABLE_MODELS['flash25lite']}\n"
        f"• flash20 → {AVAILABLE_MODELS['flash20']}\n"
        f"• flash20lite → {AVAILABLE_MODELS['flash20lite']}\n\n"
        f"Текущая модель: {current_model}"
    )

    await update.message.reply_text(text)


async def current_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    model_key = user_models.get(user_id, DEFAULT_MODEL)

    await update.message.reply_text(
        f"Текущая модель:\n"
        f"{AVAILABLE_MODELS[model_key]}"
    )


async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id

    if not context.args:
        await update.message.reply_text(
            "Укажи модель.\n\n"
            "Пример:\n"
            "/setmodel flash25\n\n"
            "Список моделей:\n"
            + '\n'.join(AVAILABLE_MODELS.keys())
        )
        return

    selected = context.args[0].lower()

    if selected not in AVAILABLE_MODELS:
        await update.message.reply_text(
            "Неизвестная модель.\n\n"
            "Доступные модели:\n"
            + '\n'.join(AVAILABLE_MODELS.keys())
        )
        return

    user_models[user_id] = selected

    await update.message.reply_text(
        f"Модель переключена на:\n"
        f"{AVAILABLE_MODELS[selected]}"
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    user_histories[user_id] = []

    await update.message.reply_text("История очищена!")


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
            f"🤖 Модель: {AVAILABLE_MODELS[current]}"
        )

        MAX_LENGTH = 4000

        for i in range(0, len(full_response), MAX_LENGTH):
            chunk = full_response[i:i + MAX_LENGTH]
            await update.message.reply_text(chunk)

    except Exception as e:
        logger.error(f"Chat error: {e}")

        await update.message.reply_text(
            "Ошибка при обращении к Gemini. Попробуй ещё раз."
        )


async def image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "Укажи описание.\nПример: /image sunset over mountains"
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
        "Генерирую изображение, подожди ~60 секунд..."
    )

    try:
        api_key = os.getenv('HORDE_API_KEY', '0000000000')

        headers = {
            'apikey': api_key,
            'Content-Type': 'application/json'
        }

        submit = requests.post(
            'https://stablehorde.net/api/v2/generate/async',
            headers=headers,
            json={
                'prompt': prompt,
                'params': {
                    'width': 512,
                    'height': 512,
                    'steps': 20,
                },
                'models': ['stable_diffusion']
            },
            timeout=30
        )

        if submit.status_code != 202:
            await msg.edit_text(
                f"Ошибка отправки задачи: {submit.status_code}"
            )
            return

        job_id = submit.json()['id']

        logger.info(f"Horde job ID: {job_id}")

        import time

        for attempt in range(24):
            time.sleep(10)

            check = requests.get(
                f'https://stablehorde.net/api/v2/generate/check/{job_id}',
                headers=headers,
                timeout=60
            )

            data = check.json()

            if data.get('done'):
                break

        result = requests.get(
            f'https://stablehorde.net/api/v2/generate/status/{job_id}',
            headers=headers,
            timeout=60
        )

        generations = result.json().get('generations', [])

        if not generations:
            await msg.edit_text(
                "Изображение не сгенерировалось. Попробуй ещё раз."
            )
            return

        image_url = generations[0]['img']

        img_response = requests.get(
            image_url,
            timeout=60
        )

        await update.message.reply_photo(
            photo=io.BytesIO(img_response.content),
            caption=full_text
        )

        await msg.delete()

    except Exception as e:
        logger.error(f"Image error: {e}")

        await msg.edit_text(f"Ошибка: {str(e)}")


def main():
    token = os.getenv('TELEGRAM_TOKEN')

    if not token:
        raise ValueError(
            "TELEGRAM_TOKEN не найден в .env файле"
        )

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("clear", clear))

    # Модели
    app.add_handler(CommandHandler("models", models))
    app.add_handler(CommandHandler("model", current_model))
    app.add_handler(CommandHandler("setmodel", set_model))

    # Картинки
    app.add_handler(CommandHandler("image", image))

    # Чат
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
