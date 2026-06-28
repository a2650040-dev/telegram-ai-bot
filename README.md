# Telegram AI Bot

A private Telegram bot powered by Google Gemini. Supports multi-turn chat with memory, image generation via Pollinations/Flux, per-user model switching, and proper Markdown rendering in Telegram. Deployed 24/7 on Railway. 100% free stack.

---

## Features

- **AI chat** — multi-turn conversation with Gemini, per-user history (last 20 messages)
- **Image generation** — Flux model via Pollinations, 5 built-in style presets
- **Model switching** — choose between 4 Gemini models per user session
- **Markdown rendering** — bold, italic, inline code, code blocks rendered natively in Telegram (custom recursive MarkdownV2 converter with fallback)
- **Whitelist** — access restricted to approved Telegram user IDs via `allowed_users.txt`

---

## Commands

| Command | Description |
|---|---|
| `[any text]` | Chat with Gemini AI (remembers context) |
| `/image [description]` | Generate an image |
| `/image [description] --style` | Generate with a style preset |
| `/model` | Show current Gemini model |
| `/models` | List all available models |
| `/setmodel [name]` | Switch Gemini model |
| `/clear` | Clear conversation history |
| `/start` | Welcome message |
| `/help` | Command reference |

**Image style presets:** `--photo` `--anime` `--art` `--dark` `--minimal`

---

## Stack

| Component | Service | Notes |
|---|---|---|
| LLM | Google Gemini API | `google-genai` SDK |
| Image generation | Pollinations.ai + Flux | `gen.pollinations.ai` endpoint |
| Hosting | Railway | 24/7, Python 3.11 |
| Framework | python-telegram-bot 22.8 | Async polling |

All services used on free tiers.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/a2650040-dev/telegram-ai-bot
cd telegram-ai-bot
pip install -r requirements.txt
```

### 2. Create `.env`

```env
TELEGRAM_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
POLLINATIONS_KEY=sk_your_pollinations_key
```

- **Telegram token** — [@BotFather](https://t.me/BotFather) → `/newbot`
- **Gemini API key** — [aistudio.google.com](https://aistudio.google.com)
- **Pollinations key** — [enter.pollinations.ai](https://enter.pollinations.ai) → Sign up → API Keys → Create (`sk_...`)

### 3. Configure whitelist

Create `allowed_users.txt` in the project root. Add one Telegram user ID per line:

```
# Find your ID by messaging @userinfobot on Telegram
123456789
987654321
```

If the file is missing, the bot is open to everyone (warning logged on startup).

### 4. Run locally

```bash
python bot.py
```

---

## Deployment (Railway)

### Prerequisites

- `Procfile` — `worker: python bot.py`
- `runtime.txt` — `python-3.11` (required — Railway defaults to Python 3.13 which breaks PTB)

### Deploy

```bash
git add .
git commit -m "initial deploy"
git push
```

Railway auto-deploys on every push to `master`.

### Environment variables

Set in Railway dashboard → Variables:

```
TELEGRAM_TOKEN
GEMINI_API_KEY
POLLINATIONS_KEY
```

> After changing variables in Railway, click **Deploy** manually — they don't apply automatically.

---

## Available Gemini Models

| Alias | Model |
|---|---|
| `flash25` *(default)* | `gemini-2.5-flash` |
| `flash25lite` | `gemini-2.5-flash-lite` |
| `flash20` | `gemini-2.0-flash` |
| `flash20lite` | `gemini-2.0-flash-lite` |

---

## Project Structure

```
telegram-ai-bot/
├── bot.py               # Main bot code
├── allowed_users.txt    # Whitelist of permitted Telegram user IDs
├── .env                 # Secrets — not committed to git
├── requirements.txt
├── Procfile             # worker: python bot.py
├── runtime.txt          # python-3.11
└── .gitignore
```

---

## Known Limitations

- Conversation history is stored in process memory — resets on every redeploy
- Bot is single-instance only — running locally while Railway is active causes a `409 Conflict` error (Telegram only allows one polling connection per token)

---

## Dependencies

```
python-telegram-bot==22.8
google-genai==2.9.0
requests
python-dotenv
httpx==0.28.1
```
