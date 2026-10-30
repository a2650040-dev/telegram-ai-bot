# -*- coding: utf-8 -*-
"""
Roles (personas) for the bot.

Each role is a system prompt kept in its own file in the `roles/` folder next
to this module (roles/orion.md, roles/martin.md, ...). Edit the text of a role
by editing its .md file - no code changes needed. To add a role, put a new
.md file into `roles/` and add one line to _ROLE_DEFS below.

The "default" role has no prompt: it is the plain bot, exactly as it worked
before roles existed. All conversation history written before roles were
introduced belongs to this role.
"""
import os
import logging

logger = logging.getLogger(__name__)

ROLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roles')

# Must match DEFAULT_ROLE_KEY in memory.py (that's the value old messages get).
DEFAULT_ROLE = 'default'

# (key, menu label, one-line description, prompt file or None)
_ROLE_DEFS = [
    ('default', '💬 Обычный бот', 'обычный помощник без роли', None),
    ('orion',   '🧭 Орион',  'друг и наставник: решения, цели, итоги дня', 'orion.md'),
    ('martin',  '📚 Мартин', 'чтение, русский язык, правка и оформление текстов', 'martin.md'),
    ('nika',    '💛 Ника',   'умная подруга, влюблена в тебя', 'nika.md'),
    ('inga',    '🧾 Инга',   'бывшая жена: прямой и жёсткий разговор', 'inga.md'),
    ('irina',   '🌿 Ирина',  'психолог', 'irina.md'),
    ('emma',    '🇬🇧 Эмма',  'преподаватель английского', 'emma.md'),
]


def load_roles(roles_dir=None):
    """Returns {key: {'label', 'description', 'prompt'}}. A role whose file is
    missing or empty is skipped with a warning (the bot still starts); the
    default role is always present."""
    directory = roles_dir or ROLES_DIR
    roles = {}
    for key, label, description, filename in _ROLE_DEFS:
        if filename is None:
            prompt = ''
        else:
            path = os.path.join(directory, filename)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    prompt = f.read().strip()
            except OSError:
                logger.warning(f"Role {key!r}: prompt file {path} not found - role skipped")
                continue
            if not prompt:
                logger.warning(f"Role {key!r}: prompt file {path} is empty - role skipped")
                continue
        roles[key] = {'label': label, 'description': description, 'prompt': prompt}
    logger.info(f"Loaded {len(roles)} role(s): {', '.join(roles)}")
    return roles


ROLES = load_roles()


def find_role_key(text):
    """Matches what the user typed (/role Орион, /role orion, /role обычный бот)
    to a role key. Returns None if nothing matches."""
    t = (text or '').strip().lower()
    if not t:
        return None
    for key, role in ROLES.items():
        name = role['label'].split(' ', 1)[-1].lower()  # label without the emoji
        if t in (key, name):
            return key
    if t in ('обычный', 'без роли', 'none', 'off'):
        return DEFAULT_ROLE
    return None
