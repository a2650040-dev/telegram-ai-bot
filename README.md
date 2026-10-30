# Telegram AI Bot

A Telegram bot powered by Google Gemini. Multi-turn chat with persistent memory (Postgres), roles (personas) with separate conversation history, voice in/out, photo and PDF understanding, optional Google Search, image generation via Pollinations, per-user model switching, and Markdown rendering in Telegram. Deployed 24/7 on Railway.

---

## Features

- **AI chat** — multi-turn conversation with Gemini. History (last 20 messages) and a long-term user profile are stored in Postgres, so they survive redeploys.
- **Roles** — besides the plain bot you can talk to a character: Орион (friend and mentor), Мартин (reading partner, Russian language and literature), Ника (smart friend in love with you), Инга (ex-wife, blunt), Ирина (psychologist), Эмма (English teacher). Choose with `/role` or `/menu` → Roles. **Every role has its own conversation history** (switch and come back, the role continues where it stopped); the user profile is shared by all roles. `/clear` clears only the current role. The role prompts are plain text files in the `roles/` folder — edit a file to change a role. The default role is the plain bot and owns all history written before roles existed.
- **Photos and PDFs** — send a photo, an image file (JPG, PNG, WebP, HEIC) or a PDF. A caption works as the question; without one the bot describes the image / summarizes the PDF. Max 15 MB. Only a short text note is stored in memory, not the file.
- **Google Search** — optional grounding with Google Search. Turn it on for every message with `/search`, or just ask in chat ("поищи в интернете …", "search the web for …") to search for that one message. If search is not allowed for the API key (e.g. free tier), the bot answers without it and says so. Not used for voice messages.
- **Date and time** — the model is told the current date/time (time zone from `BOT_TIMEZONE`, default `Asia/Yakutsk`).
- **Voice** — voice messages are understood; replies can be text or voice (see `/menu` → Voice). A "Read aloud" button under replies reads them with Gemini TTS.
- **Image generation** — Pollinations (`gen.pollinations.ai`), style presets.
- **Model switching** — per-user model choice (saved in Postgres). Some models are for trusted users only.
- **Markdown rendering** — custom MarkdownV2 converter with plain-text fallback.
- **Access control** — `allowed_users.txt` lists trusted users; with `PUBLIC_MODE=true` everyone can chat but only trusted users get the higher-tier models. (Roles are available to everyone who can chat.)
- **Rate limits** (in memory, reset on restart) — 20 chat messages per hour and 70 per day per user, 10 images per day per user, 400 requests per day for the whole bot.

---

## Commands

| Command | Description |
|---|---|
| `[any text]` | Chat with Gemini (remembers context) |
| photo / PDF | Ask about it (caption = question) |
| `/menu` | Button menu |
| `/welcome` | Welcome / help text |
| `/role [name]` | Choose a role (no argument shows the list; `/role Орион`, `/role обычный бот`) |
| `/image [description]` | Generate an image (legacy flags: `--photo --anime --art --dark --minimal`) |
| `/model`, `/models`, `/setmodel [name]` | Show / list / switch model |
| `/search [on\|off]` | Turn Google Search on or off (no argument toggles) |
| `/clear` | Clear the conversation history of the **current role** (keeps other roles, profile and settings) |
| `/undo` | Remove the last exchange of the current role |
| `/privacy` | What the bot remembers |
| `/forget_me` | Delete all stored data about you (all roles) |
| `/stats` | Usage statistics (admin only) |

---

## Stack

| Component | Service | Notes |
|---|---|---|
| LLM | Google Gemini API | `google-genai` SDK |
| Memory | Postgres (Railway) | `asyncpg`, see `memory.py` |
| Image generation | Pollinations.ai | `gen.pollinations.ai` endpoint |
| Voice replies | Gemini TTS + `pydub` | `pydub` needs **ffmpeg** on the host |
| Hosting | Railway | worker process, Python 3.11 |
| Framework | python-telegram-bot 22.8 | async polling |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/a2650040-dev/telegram-ai-bot
cd telegram-ai-bot
pip install -r requirements.txt
```

### 2. Environment variables (`.env` locally, Railway → Variables in production)

```env
TELEGRAM_TOKEN=your_telegram_bot_token
GEMINI_API_KEY=your_gemini_api_key
POLLINATIONS_KEY=your_pollinations_key
DATABASE_URL=postgres://...
ADMIN_USER_ID=your_telegram_user_id
PUBLIC_MODE=false
BOT_TIMEZONE=Asia/Yakutsk
```

- **Telegram token** — [@BotFather](https://t.me/BotFather)
- **Gemini API key** — [aistudio.google.com](https://aistudio.google.com)
- **Pollinations key** — [enter.pollinations.ai](https://enter.pollinations.ai)
- `DATABASE_URL` — the Postgres connection string (on Railway, reference the Postgres service)
- `ADMIN_USER_ID` — your Telegram ID, enables `/stats`
- `PUBLIC_MODE` — `true` lets anyone talk to the bot
- `BOT_TIMEZONE` — optional, an IANA name such as `Europe/Berlin`; default `Asia/Yakutsk`

### 3. Trusted users

Create `allowed_users.txt` in the project root, one Telegram user ID per line (`#` comments allowed):

```
# Find your ID by messaging @userinfobot on Telegram
123456789
```

If the file is missing, nobody is trusted (a warning is logged on startup).

### 4. Run locally

```bash
python bot.py
```

---

## Roles

Each role is one text file in `roles/` (`orion.md`, `martin.md`, `nika.md`, `inga.md`, `irina.md`, `emma.md`). The list of roles (key, menu label, one-line description, file) is `_ROLE_DEFS` in `roles.py`. To change a role, edit its `.md` file and redeploy. To add a role, put a new `.md` file into `roles/` and add one line to `_ROLE_DEFS`. If a role's file is missing or empty, that role is skipped (a warning is logged) and the bot still starts.

How memory works with roles:

- messages are stored with a `role_key` (`messages.role_key`, old rows = `default`); the current role is in `user_profile.role_key`;
- history in RAM is kept per `(user_id, role_key)`;
- the profile summary is shared by all roles and is built from the recent messages of all roles (the profile prompt tells the model to record only facts the user stated).

---

## Deployment (Railway)

- `Procfile` — `worker: python bot.py`
- `runtime.txt` — `python-3.11` (Railway defaults to a newer Python that breaks PTB)

```bash
git add .
git commit -m "message"
git push
```

Railway deploys on every push to `master`. After changing variables in Railway, deploy again so they apply. The database is migrated automatically on startup (new columns are added with `ADD COLUMN IF NOT EXISTS`).

---

## Available Gemini models (as of October 2026)

| Alias | Model | Access |
|---|---|---|
| `flash35lite` *(default)* | `gemini-3.5-flash-lite` | everyone |
| `flash31lite` | `gemini-3.1-flash-lite` | everyone |
| `flash38` | `gemini-3.8-flash` | trusted users |
| `flash36` | `gemini-3.6-flash` | trusted users |

Text-to-speech uses `gemini-3.8-flash-lite-tts`. Gemini 2.x models were removed (legacy on the Gemini API).

---

## Project Structure

```
telegram-ai-bot/
├── bot.py               # Main bot code
├── memory.py            # Postgres memory: history (per role), profile, per-user settings
├── roles.py             # Loads the role prompts, role list
├── roles/               # One .md prompt per role (orion, martin, nika, inga, irina, emma)
├── allowed_users.txt    # Trusted Telegram user IDs
├── claude/              # Notes (TECH_DEBT.md) and smoke_test.py
├── .env                 # Secrets — not committed to git
├── requirements.txt
├── Procfile             # worker: python bot.py
├── runtime.txt          # python-3.11
└── .gitignore
```

---

## Known Limitations

- Bot is single-instance only — running locally while Railway is active causes a `409 Conflict` error (Telegram allows one polling connection per token).
- Rate limits live in memory and reset on every restart.
- Some calls to Gemini and Pollinations are synchronous, so replies are handled one after another.
- Google Search grounding may require a paid Gemini API tier; the bot falls back to answering without it.
- Roles are plain system prompts: the model can drift from a character in a long conversation, and the history window is 20 messages per role (better per-role memory is planned).

---

## Dependencies

```
python-telegram-bot==22.8
google-genai==2.9.0
requests
python-dotenv
httpx==0.28.1
pydub
asyncpg
tzdata
```
