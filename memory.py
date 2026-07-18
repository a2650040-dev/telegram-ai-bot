# -*- coding: utf-8 -*-
"""
Persistent memory for the bot: recent conversation history + a long-term
user profile, stored in Postgres (Railway addon). This is what makes
memory survive deploys/restarts instead of living only in RAM.
"""
import os
import logging

import asyncpg

logger = logging.getLogger(__name__)

_pool = None


async def get_pool():
    global _pool
    if _pool is None:
        database_url = os.environ["DATABASE_URL"]
        _pool = await asyncpg.create_pool(database_url, ssl="require")
    return _pool


async def init_db():
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT now()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id BIGINT PRIMARY KEY,
                summary TEXT DEFAULT '',
                message_count INT DEFAULT 0,
                updated_at TIMESTAMPTZ DEFAULT now()
            );
        """)
        # Added later: per-user settings that used to live only in RAM.
        # ADD COLUMN IF NOT EXISTS is safe to re-run on every startup, so
        # this also upgrades any user_profile table created before this.
        await conn.execute("ALTER TABLE user_profile ADD COLUMN IF NOT EXISTS model_key TEXT;")
        await conn.execute("ALTER TABLE user_profile ADD COLUMN IF NOT EXISTS voice_mode TEXT;")
    logger.info("Memory DB ready (messages, user_profile)")


async def save_message(user_id: int, role: str, content: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO messages (user_id, role, content) VALUES ($1, $2, $3)",
            user_id, role, content,
        )
        await conn.execute(
            """
            INSERT INTO user_profile (user_id, message_count)
            VALUES ($1, 1)
            ON CONFLICT (user_id) DO UPDATE
            SET message_count = user_profile.message_count + 1
            """,
            user_id,
        )


async def get_recent_history(user_id: int, limit: int = 20):
    """Returns rows (role, content) in chronological order."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content FROM messages
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            user_id, limit,
        )
        return list(reversed(rows))


async def clear_history(user_id: int):
    """Wipes recent conversation history. Deliberately does NOT touch the
    long-term profile - /clear is about the current conversation, not
    about the bot forgetting who the user is."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM messages WHERE user_id = $1", user_id)


async def get_profile(user_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT summary, message_count FROM user_profile WHERE user_id = $1",
            user_id,
        )


async def get_user_settings(user_id: int):
    """Returns a row with model_key and voice_mode (either may be None if
    the user never set them), or None if the user has no profile row yet."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT model_key, voice_mode FROM user_profile WHERE user_id = $1",
            user_id,
        )


async def set_user_model(user_id: int, model_key: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_profile (user_id, model_key)
            VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET model_key = $2
            """,
            user_id, model_key,
        )


async def set_user_voice_mode(user_id: int, voice_mode: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_profile (user_id, voice_mode)
            VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET voice_mode = $2
            """,
            user_id, voice_mode,
        )


async def update_summary(user_id: int, new_summary: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE user_profile SET summary = $2, updated_at = now() WHERE user_id = $1",
            user_id, new_summary,
        )


async def maybe_refresh_summary(user_id: int, gemini_client, model_name: str = "gemini-2.5-flash"):
    """Every 20 messages, asks Gemini to distill a compact, durable profile
    from recent history - keeps memory useful without growing forever."""
    profile = await get_profile(user_id)
    if not profile or profile["message_count"] == 0 or profile["message_count"] % 20 != 0:
        return

    recent = await get_recent_history(user_id, limit=30)
    convo_text = "\n".join(f"{r['role']}: {r['content']}" for r in recent)

    prompt = f"""Current profile notes about this user:
{profile['summary'] or '(empty)'}

Recent conversation:
{convo_text}

Update the profile: short bullet points, only durable facts (name,
interests, communication style, preferences). Do not include one-off
details specific to this conversation. Reply with only the updated
profile text, no preamble."""

    try:
        response = gemini_client.models.generate_content(model=model_name, contents=prompt)
        await update_summary(user_id, response.text.strip())
        logger.info(f"Refreshed profile summary for user_id={user_id}")
    except Exception:
        logger.exception(f"Failed to refresh profile summary for user_id={user_id}")
