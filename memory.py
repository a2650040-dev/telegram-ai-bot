# -*- coding: utf-8 -*-
"""
Persistent memory for the bot: conversation history (kept separately for
every role) + a long-term user profile shared by all roles, stored in
Postgres (Railway addon). This is what makes memory survive
deploys/restarts instead of living only in RAM.
"""
import os
import logging

import asyncpg

logger = logging.getLogger(__name__)

# Role that old messages (written before roles existed) belong to.
# Must match DEFAULT_ROLE in roles.py.
DEFAULT_ROLE_KEY = 'default'

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
        await conn.execute("ALTER TABLE user_profile ADD COLUMN IF NOT EXISTS search_mode TEXT;")
        # Roles: the role the user is currently talking to (NULL = default),
        # and which role every stored message belongs to. Rows that existed
        # before this migration automatically get role_key = 'default', so the
        # whole old history becomes the history of the default role.
        await conn.execute("ALTER TABLE user_profile ADD COLUMN IF NOT EXISTS role_key TEXT;")
        await conn.execute(
            f"ALTER TABLE messages ADD COLUMN IF NOT EXISTS role_key TEXT NOT NULL DEFAULT '{DEFAULT_ROLE_KEY}';"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS messages_user_role_id_idx ON messages (user_id, role_key, id);"
        )
    logger.info("Memory DB ready (messages, user_profile)")


async def save_message(user_id: int, role: str, content: str, role_key: str = DEFAULT_ROLE_KEY):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO messages (user_id, role, content, role_key) VALUES ($1, $2, $3, $4)",
            user_id, role, content, role_key,
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


async def get_recent_history(user_id: int, limit: int = 20, role_key: str | None = None):
    """Returns rows (role, content) in chronological order. With role_key,
    only that role's conversation; with role_key=None, all roles together
    (used only to refresh the shared profile)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        if role_key is None:
            rows = await conn.fetch(
                """
                SELECT role, content FROM messages
                WHERE user_id = $1
                ORDER BY id DESC
                LIMIT $2
                """,
                user_id, limit,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT role, content FROM messages
                WHERE user_id = $1 AND role_key = $2
                ORDER BY id DESC
                LIMIT $3
                """,
                user_id, role_key, limit,
            )
        return list(reversed(rows))


async def clear_history(user_id: int, role_key: str):
    """Wipes the conversation history of ONE role. Deliberately does NOT touch
    other roles' history or the long-term profile - /clear is about the
    current conversation, not about the bot forgetting who the user is."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM messages WHERE user_id = $1 AND role_key = $2",
            user_id, role_key,
        )


async def delete_last_messages(user_id: int, count: int = 2, role_key: str = DEFAULT_ROLE_KEY):
    """Deletes the most recent `count` message rows of one role - used by
    /undo to recover from a broken/blocked turn without wiping the whole
    history like /clear does."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            DELETE FROM messages WHERE id IN (
                SELECT id FROM messages WHERE user_id = $1 AND role_key = $2
                ORDER BY id DESC LIMIT $3
            )
            """,
            user_id, role_key, count,
        )


async def delete_user_data(user_id: int):
    """Full wipe: history of ALL roles, profile summary, and saved settings -
    everything Postgres knows about this user. Used by /forget_me, unlike
    /clear which only touches one role's conversation history."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM messages WHERE user_id = $1", user_id)
        await conn.execute("DELETE FROM user_profile WHERE user_id = $1", user_id)


async def get_profile(user_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT summary, message_count FROM user_profile WHERE user_id = $1",
            user_id,
        )


async def get_user_settings(user_id: int):
    """Returns a row with model_key, voice_mode, search_mode and role_key (any
    may be None if the user never set them), or None if the user has no
    profile row yet."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(
            "SELECT model_key, voice_mode, search_mode, role_key FROM user_profile WHERE user_id = $1",
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


async def set_user_search_mode(user_id: int, search_mode: str):
    """search_mode is 'on' or 'off' (Google Search grounding for chat replies)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_profile (user_id, search_mode)
            VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET search_mode = $2
            """,
            user_id, search_mode,
        )


async def set_user_role(user_id: int, role_key: str):
    """Remembers which role the user is talking to, so it survives restarts."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_profile (user_id, role_key)
            VALUES ($1, $2)
            ON CONFLICT (user_id) DO UPDATE SET role_key = $2
            """,
            user_id, role_key,
        )


async def update_summary(user_id: int, new_summary: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE user_profile SET summary = $2, updated_at = now() WHERE user_id = $1",
            user_id, new_summary,
        )


async def maybe_refresh_summary(user_id: int, gemini_client, model_name: str = "gemini-3.5-flash-lite"):
    """Every 20 messages, asks Gemini to distill a compact, durable profile
    from recent history - keeps memory useful without growing forever.
    The profile is shared by all roles, so it is built from the recent
    messages of all roles together."""
    profile = await get_profile(user_id)
    if not profile or profile["message_count"] == 0 or profile["message_count"] % 20 != 0:
        return

    recent = await get_recent_history(user_id, limit=30)
    convo_text = "\n".join(f"{r['role']}: {r['content']}" for r in recent)

    prompt = f"""Current profile notes about this user:
{profile['summary'] or '(empty)'}

Recent conversation (the user may be talking to different assistant
personas, and the assistant's lines may be written in character):
{convo_text}

Update the profile: short bullet points, only durable facts the USER told
about themselves (name, interests, communication style, preferences).
Never record anything a persona said, invented or implied about itself or
about the user. Do not include one-off details specific to this
conversation. Reply with only the updated profile text, no preamble."""

    try:
        response = gemini_client.models.generate_content(model=model_name, contents=prompt)
        await update_summary(user_id, response.text.strip())
        logger.info(f"Refreshed profile summary for user_id={user_id}")
    except Exception:
        logger.exception(f"Failed to refresh profile summary for user_id={user_id}")
