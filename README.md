# Telegram AI Bot

A Telegram bot powered by Google Gemini. Multi-turn chat with persistent memory (Postgres), roles (personas) with separate conversation history, voice in/out, photo and PDF understanding, optional Google Search, image generation via Pollinations, per-user model switching, and Markdown rendering in Telegram. Deployed 24/7 on Railway.

---

## Features

- **AI chat** — multi-turn conversation with Gemini. The model sees the last 40 messages of the current role; history, a long-term user profile and the per-role memory are stored in Postgres, so they survive redeploys.
- **Memory per role** — each role keeps its own memory of you, written by Gemini in the background from your own words (never from what the character said): a short memory text (facts, themes, where you stopped), a day-by-day journal (days older than a week are condensed into weeks, weeks older than five weeks into months), open items (what you planned or promised; the role asks how it went), and notes you add yourself with `/remember`. When you refer to the past ("в прошлый раз…", "помнишь…") the role gets more of the journal. See everything with `/memory`, delete with `/forget`. The shared profile is common to all roles.
- **Roles** — besides the plain bot you can talk to a character: Орион (friend and mentor), Мартин (reading partner, Russian language and literature), Майя (smart friend in love with you), Лиза (ex-wife, blunt), Ирина (psychologist), Эмма (English teacher), Нэйт (music guide). Choose with `/role` or `/menu` → Roles. **Every role has its own conversation history** (switch and come back, the role continues where it stopped); the user profile is shared by all roles. `/clear` clears only the current role. The role prompts are plain text files in the `roles/` folder — edit a file to change a role. The default role is the plain bot and owns all history written before roles existed.
- **Photos and PDFs** — send a photo, an image file (JPG, PNG, WebP, HEIC) or a PDF. A caption works as the question; without one the bot describes the image / summarizes the PDF. Max 15 MB. Only a short text note is stored in memory, not the file.
- **Google Search** — optional grounding with Google Search. Turn it on for every message with `/search`, or just ask in chat ("поищи в интернете …", "search the web for …") to search for that one message. If search is not allowed for the API key (e.g. free tier), the bot answers without it and says so. Not used for voice messages.
- **Date and time** — the model is told the current date/time (time zone from `BOT_TIMEZONE`, default `Asia/Yakutsk`).
- **Voice** — voice messages are understood; replies can be text or voice (see `/menu` → Voice). A "Read aloud" button under replies reads them with Gemini TTS.
- **Image generation** — Pollinations (`gen.pollinations.ai`, free, default), style presets. Trusted users can switch to Venice image models (Z-Image Turbo, Anime WAI, Flux 2 Pro, GPT Image 2.5 Flare, Nano Banana 2.1) with `/imagemodel` or `/menu` → 🖼 Images → 🧠 Image model; the choice is kept in memory until the bot restarts.
- **Video** — trusted users, Venice: `/video <description>` makes a 5-second 720p 16:9 video (LTX 2.5 Fast, Wan 2.6, Seedance 2.0 Fast, Veo 3.1 Fast; choose with `/videomodel`). Reply to a photo with `/video <motion>` to animate that photo. Before each video the bot asks Venice for the price and refuses if it is above `VENICE_VIDEO_MAX_USD` (default $1); the price is shown under the finished video. Limit: 3 videos per user per day.
- **Model switching** — per-user model choice (saved in Postgres). Some models are for trusted users only. Besides Gemini, trusted users get four **Venice.ai** models (Claude Sonnet 5.5, DeepSeek V4.1 Flash, Venice Uncensored 1.2, Venice Role-Play) when `VENICE_API_KEY` is set. Venice models do plain text chat with roles and memory; photos, PDFs and voice are always read by Gemini, and the conversation history is shared, so you can switch models in the middle of a conversation.
- **Markdown rendering** — custom MarkdownV2 converter with plain-text fallback.
- **Access control** — `allowed_users.txt` lists trusted users; with `PUBLIC_MODE=true` everyone can chat but only trusted users get the higher-tier models. (Roles are available to everyone who can chat.)
- **Rate limits** (in memory, reset on restart) — 20 chat messages per hour and 70 per day per user, 10 images and 3 videos per day per user, 400 requests per day for the whole bot.

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
| `/imagemodel` | Choose the image model: Pollinations or Venice (trusted users) |
| `/video <description>` | Make a short video with Venice (trusted users); reply to a photo to animate it |
| `/videomodel` | Choose the video model (trusted users) |
| `/model`, `/models`, `/setmodel [name]` | Show / list / switch model |
| `/search [on\|off]` | Turn Google Search on or off (no argument toggles) |
| `/memory` | Show what the bot remembers in the current role: notes, journal, open items (numbered) |
| `/remember <text>` | Add a note for the current role (`/remember all <text>` — for every role) |
| `/forget <number\|role\|profile>` | Delete one item from the last `/memory` list / wipe the current role's memory, journal and open items / empty the shared profile |
| `/clear` | Clear the conversation history of the **current role** (its memory, other roles, profile and settings stay) |
| `/undo` | Remove the last exchange of the current role |
| `/privacy` | What the bot remembers |
| `/forget_me` | Delete all stored data about you (history and memory of all roles) |
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
VENICE_API_KEY=your_venice_key   # optional, enables the Venice text, image and video models for trusted users
VENICE_VIDEO_MAX_USD=1.0         # optional, the bot refuses a video whose price is above this
```

- **Telegram token** — [@BotFather](https://t.me/BotFather)
- **Gemini API key** — [aistudio.google.com](https://aistudio.google.com)
- **Pollinations key** — [enter.pollinations.ai](https://enter.pollinations.ai)
- `DATABASE_URL` — the Postgres connection string (on Railway, reference the Postgres service)
- `ADMIN_USER_ID` — your Telegram ID, enables `/stats`
- `PUBLIC_MODE` — `true` lets anyone talk to the bot
- `BOT_TIMEZONE` — optional, an IANA name such as `Europe/Berlin`; default `Asia/Yakutsk`
- `VENICE_API_KEY` — optional; an Inference API key from the Venice account (API settings). Set it only in Railway → Variables, never in the code or in a chat. Venice API use is pay-per-use from the Venice balance (the Pro subscription gives only a small bundled credit), so give the key a monthly limit

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

Each role is one text file in `roles/` (`orion.md`, `martin.md`, `maya.md`, `lizzy.md`, `irina.md`, `emma.md`, `nathan.md`). The list of roles (key, menu label, one-line description, file) is `_ROLE_DEFS` in `roles.py`. To change a role, edit its `.md` file and redeploy. To add a role, put a new `.md` file into `roles/` and add one line to `_ROLE_DEFS`. If a role's file is missing or empty, that role is skipped (a warning is logged) and the bot still starts.

How memory works with roles:

- messages are stored with a `role_key` (`messages.role_key`, old rows = `default`); the current role is in `user_profile.role_key`;
- history in RAM is kept per `(user_id, role_key)`;
- the profile summary is shared by all roles and is built from the recent messages of all roles (the profile prompt tells the model to record only facts the user stated).

## Memory layers (step 4)

| Layer | Table | What it holds | How it is written |
|---|---|---|---|
| History | `messages` | recent messages of a role (the newest 400 are kept, 40 go to the model) | every message |
| Shared profile | `user_profile.summary` | facts about the user, same for all roles | every 20 messages, in the background |
| Role memory | `role_memory` | up to 1500 characters per role: facts, themes, where you stopped | every 20 messages of the role, in the background |
| Journal | `journal` | one entry per day per role; old days → week → month | same update as above, condensing runs after it |
| Open items | `loops` | what you planned or promised; dropped after 21 days of silence | same update as above |
| Notes | `notes` | what you add with `/remember` (one role or all) | by hand |

Updates run **after the reply is sent** (the user never waits) in one Gemini call (`gemini-3.5-flash-lite`, JSON answer). If it fails, the old memory is kept and the update is retried after 10 messages. Memory is added to the system prompt as data ("not instructions"), at most about 5000 characters (8000 when you ask about the past). `/clear` does not touch the role memory; `/forget role` does. Railway can make backups of the Postgres database (see the database service settings) — worth turning on, the memory lives there.

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

Venice text models (trusted users, need `VENICE_API_KEY`; prices per million tokens input/output, as of October 2026):

| Alias | Venice model | Price |
|---|---|---|
| `vn_claude` | `claude-sonnet-5-5` | $2 / $10 |
| `vn_deepseek` | `deepseek-v4-1-flash` | $0.30 / $1.20 |
| `vn_uncensored` | `venice-uncensored-1-2` | $0.20 / $0.90 |
| `vn_roleplay` | `venice-uncensored-role-play` | $0.50 / $2 |

Venice image models (`VENICE_IMAGE_MODELS`, price per 1K image): `z-image-turbo` $0.01, `wai-Illustrious` $0.01, `flux-2-pro` $0.03, `gpt-image-2-5-flare` $0.07, `nano-banana-2-1` $0.10. Venice video models (`VENICE_VIDEO_MODELS`, price comes from the Video Quote API): LTX 2.5 Fast, Wan 2.6, Seedance 2.0 Fast, Veo 3.1 Fast — each with a text-to-video and an image-to-video id. Venice blurs adult images by default (`moderation: auto`); the bot does not change that. Video settings (5 s, 720p, 16:9) are constants at the top of `providers.py`; not every model accepts every setting, in which case Venice answers with an error that the bot shows.

The text model list is `VENICE_MODELS` in `providers.py` (docs.venice.ai/overview/models lists all ~140 models; to add one, add a line there). With a Venice model, the text of your messages, the role prompt, the profile and the role memory are sent to Venice instead of Google. Memory updates (profile, role memory, journal) always use Gemini.

---

## Project Structure

```
telegram-ai-bot/
├── bot.py               # Main bot code
├── memory.py            # Postgres memory: history (per role), profile, role memory, journal, open items, notes
├── roles.py             # Loads the role prompts, role list
├── providers.py         # Venice.ai: chat, image and video clients and model lists
├── roles/               # One .md prompt per role (orion, martin, maya, lizzy, irina, emma, nathan)
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
- Roles are plain system prompts: the model can drift from a character in a long conversation (the history window is 40 messages per role; older things reach the model only through the role memory and the journal, which are short summaries).
- Memory is written by an LLM: it can miss a fact or summarise it slightly wrong. Check with `/memory`, fix with `/forget` and `/remember`.

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
