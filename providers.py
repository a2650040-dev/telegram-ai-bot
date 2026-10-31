# -*- coding: utf-8 -*-
"""
Model providers other than Gemini.

For now: Venice.ai (https://docs.venice.ai), whose API is OpenAI-compatible.
It gives the bot text chat models, image models and video models.
The key is read from the environment variable VENICE_API_KEY (set it in
Railway -> Variables; never put it into the code or the chat). Without the
key no Venice model appears in the bot.

Photos, PDFs and voice messages that the user SENDS are always read by Gemini
(see bot.py); Venice is used for text replies and for generating images/videos.
"""
import os
import re
import base64
import asyncio
import logging

import requests

logger = logging.getLogger(__name__)

VENICE_BASE_URL = 'https://api.venice.ai/api/v1'

# ── Catalogue (docs.venice.ai/overview/models, October 2026; the list changes, edit here) ──

# Text chat. Key used inside the bot -> (Venice model id, label). Price per million tokens in/out.
VENICE_MODELS = {
    'vn_uncensored': ('venice-uncensored-1-2',        'Venice Uncensored 1.2'),       # 0.20 / 0.90
    'vn_roleplay':   ('venice-uncensored-role-play',  'Venice Role-Play'),            # 0.50 / 2.00
    'vn_claude':     ('claude-sonnet-5-5',            'Claude Sonnet 5.5 (Venice)'),  # 2 / 10
    'vn_deepseek':   ('deepseek-v4-1-flash',          'DeepSeek V4.1 Flash (Venice)'),  # 0.30 / 1.20
}

# Images. Key -> (Venice model id, label). Price is per image (1K).
VENICE_IMAGE_MODELS = {
    'vi_zimage':   ('z-image-turbo',          'Z-Image Turbo ($0.01)'),
    'vi_anime':    ('wai-Illustrious',        'Anime WAI ($0.01)'),
    'vi_flux2':    ('flux-2-pro',             'Flux 2 Pro ($0.03)'),
    'vi_gptflare': ('gpt-image-2-5-flare',    'GPT Image 2.5 Flare ($0.07)'),
    'vi_banana':   ('nano-banana-2-1',        'Nano Banana 2.1 ($0.10)'),
}

# Video. Key -> {label, t2v: text-to-video model id, i2v: image-to-video model id}.
# The price depends on duration and resolution and is asked from Venice before every video.
VENICE_VIDEO_MODELS = {
    'vv_ltx':      {'label': 'LTX 2.5 Fast',    't2v': 'ltx-2-5-fast-text-to-video',
                    'i2v': 'ltx-2-5-fast-image-to-video'},
    'vv_wan':      {'label': 'Wan 2.6',         't2v': 'wan-2.6-text-to-video',
                    'i2v': 'wan-2.6-image-to-video'},
    'vv_seedance': {'label': 'Seedance 2.0 Fast', 't2v': 'seedance-2-0-fast-text-to-video-basic',
                    'i2v': 'seedance-2-0-fast-image-to-video-basic'},
    'vv_veo':      {'label': 'Veo 3.1 Fast',    't2v': 'veo3.1-fast-text-to-video',
                    'i2v': 'veo3.1-fast-image-to-video'},
}

IMAGE_PROMPT_MAX = 1500        # characters, Venice limit for /images/generations
VIDEO_DURATION = '5s'
VIDEO_RESOLUTION = '720p'
VIDEO_ASPECT = '16:9'


class ProviderError(Exception):
    """A provider request failed. user_message is safe to show in the chat
    (never contains the API key)."""
    def __init__(self, user_message, detail=''):
        super().__init__(detail or user_message)
        self.user_message = user_message


def venice_enabled() -> bool:
    return bool(os.getenv('VENICE_API_KEY'))


def video_max_usd() -> float:
    """Upper limit of the price of one video (VENICE_VIDEO_MAX_USD, default $1)."""
    try:
        return float(os.getenv('VENICE_VIDEO_MAX_USD', '1.0'))
    except ValueError:
        return 1.0


# ── HTTP layer ────────────────────────────────────────────────────────────

def _request(api_key, method, path, payload, timeout):
    url = f"{VENICE_BASE_URL}{path}"
    if method == 'GET':
        return requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
    return requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )


def _snippet(response, limit=200) -> str:
    return re.sub(r'\s+', ' ', (response.text or ''))[:limit]


async def _call(path, payload, timeout=120, max_attempts=3, refused_hint="Try another model."):
    """POST to Venice with retries on timeouts, 429 and 5xx. Returns the 200
    response. Raises ProviderError with a message safe to show to the user."""
    api_key = os.getenv('VENICE_API_KEY')
    if not api_key:
        raise ProviderError("Venice is not set up (VENICE_API_KEY is missing).")

    last_problem = "no answer"
    for attempt in range(1, max_attempts + 1):
        try:
            response = await asyncio.to_thread(_request, api_key, 'POST', path, payload, timeout)
        except (requests.Timeout, requests.ConnectionError) as e:
            last_problem = type(e).__name__
            logger.warning(f"Venice {path} failed ({last_problem}), attempt {attempt}/{max_attempts}")
            if attempt < max_attempts:
                await asyncio.sleep(2 * attempt)
                continue
            raise ProviderError("Couldn't reach Venice. Please try again in a minute.", last_problem)

        status = response.status_code
        if status == 200:
            return response

        body = (response.text or '')[:300]
        if status == 402:
            raise ProviderError("Venice balance is empty or the key's monthly limit is used up. "
                                "Top it up in the Venice account.", "402")
        if status == 401:
            if 'PRO_ONLY_MODEL' in body:
                raise ProviderError("This Venice model is only available on a paid Venice plan.", "401 pro only")
            raise ProviderError("Venice rejected the API key. Check VENICE_API_KEY in Railway.", "401")
        if status == 422:
            raise ProviderError("Venice refused this request (content rules or unsuitable settings). "
                                "Try rephrasing the description.", f"422 {_snippet(response)}")
        if status == 429 or status >= 500:
            last_problem = f"HTTP {status}"
            logger.warning(f"Venice {path} returned {status}, attempt {attempt}/{max_attempts}")
            if attempt < max_attempts:
                await asyncio.sleep(2 * attempt)
                continue
            raise ProviderError("Venice is busy or overloaded right now. Please try again in a minute.", last_problem)

        logger.warning(f"Venice {path} returned {status}: {body}")
        detail = _snippet(response)
        raise ProviderError(f"Venice refused the request (HTTP {status}). {refused_hint}"
                            + (f"\nVenice says: {detail}" if detail else ""), f"HTTP {status}")

    raise ProviderError("Venice didn't answer. Please try again.", last_problem)


# ── Text chat ─────────────────────────────────────────────────────────────

def _extract_text(data) -> str:
    """Pulls the reply out of an OpenAI-style response. Some reasoning models
    put their thoughts inline in <think> tags: those are removed."""
    try:
        content = data['choices'][0]['message'].get('content') or ''
    except (KeyError, IndexError, TypeError, AttributeError):
        return ''
    if not isinstance(content, str):
        return ''
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    return content.strip()


async def venice_chat(model: str, messages, use_search: bool = False,
                      max_attempts: int = 3, timeout: int = 120) -> str:
    """Sends the conversation to Venice and returns the reply text ('' if the
    model answered with nothing)."""
    # Venice adds its own system prompt by default; the bot's role prompt must be the only one.
    venice_parameters = {"include_venice_system_prompt": False}
    if use_search:
        venice_parameters["enable_web_search"] = "auto"
    payload = {"model": model, "messages": messages, "venice_parameters": venice_parameters}

    response = await _call('/chat/completions', payload, timeout, max_attempts,
                           refused_hint="Try another model with /setmodel.")
    try:
        return _extract_text(response.json())
    except ValueError:
        raise ProviderError("Venice sent an unreadable answer. Please try again.", "bad json")


# ── Images ────────────────────────────────────────────────────────────────

async def venice_image(model: str, prompt: str, size: str = '1024x1024') -> bytes:
    """Generates one picture and returns it as JPEG bytes."""
    prompt = (prompt or '').strip()
    if not prompt:
        raise ProviderError("Describe what to draw.")
    if len(prompt) > IMAGE_PROMPT_MAX:
        raise ProviderError(f"The description is too long for this model ({len(prompt)} characters, "
                            f"the limit is {IMAGE_PROMPT_MAX}).")
    payload = {"model": model, "prompt": prompt, "n": 1, "size": size,
               "response_format": "b64_json", "output_format": "jpeg"}
    response = await _call('/images/generations', payload, timeout=180,
                           refused_hint="Try another image model with /imagemodel.")
    try:
        return base64.b64decode(response.json()['data'][0]['b64_json'])
    except (KeyError, IndexError, TypeError, ValueError):
        raise ProviderError("Venice sent no picture. Please try again.", "no image in answer")


# ── Video ─────────────────────────────────────────────────────────────────

async def venice_video_quote(model: str, duration: str, resolution: str, aspect_ratio: str) -> float:
    response = await _call('/video/quote', {"model": model, "duration": duration,
                                            "resolution": resolution, "aspect_ratio": aspect_ratio},
                           timeout=60, refused_hint="Try another video model with /videomodel.")
    try:
        return float(response.json()['quote'])
    except (KeyError, TypeError, ValueError):
        raise ProviderError("Venice didn't give a price for this video. Please try again.", "no quote")


def _download(url, timeout):
    return requests.get(url, timeout=timeout)    # the download link needs no authorization


async def venice_video_generate(model: str, prompt: str, image_data_url: str | None = None,
                                duration: str = VIDEO_DURATION, resolution: str = VIDEO_RESOLUTION,
                                aspect_ratio: str = VIDEO_ASPECT, max_usd: float = 1.0,
                                poll_every: float = 5, max_wait: float = 900, on_status=None):
    """Quote -> queue -> poll -> download. Returns (mp4 bytes, price in USD).
    Refuses (ProviderError) when the quote is above max_usd, before spending anything.
    on_status is an optional async function(text) for progress messages."""
    prompt = (prompt or '').strip()
    if not prompt:
        raise ProviderError("Describe the video.")

    quote = await venice_video_quote(model, duration, resolution, aspect_ratio)
    if quote > max_usd:
        raise ProviderError(f"This video would cost about ${quote:.2f}, above the limit of ${max_usd:.2f}. "
                            "Pick a cheaper video model with /videomodel.")
    if on_status:
        await on_status(f"🎬 Making the video (about ${quote:.2f}). It usually takes a few minutes...")

    payload = {"model": model, "prompt": prompt, "duration": duration,
               "resolution": resolution, "aspect_ratio": aspect_ratio}
    if image_data_url:
        payload["image_url"] = image_data_url
    queued = (await _call('/video/queue', payload, timeout=180,
                          refused_hint="Try another video model with /videomodel.")).json()
    try:
        queue_id = queued['queue_id']
    except (KeyError, TypeError):
        raise ProviderError("Venice didn't accept the video job. Please try again.", "no queue_id")
    download_url = queued.get('download_url') if isinstance(queued, dict) else None
    logger.info(f"Venice video queued: model={model}, queue_id={queue_id}, quote=${quote:.2f}")

    video = None
    waited = 0.0
    polls = 0
    while waited < max_wait:
        await asyncio.sleep(poll_every)
        waited += poll_every
        polls += 1
        response = await _call('/video/retrieve', {"model": model, "queue_id": queue_id}, timeout=120)
        content_type = ((getattr(response, 'headers', None) or {}).get('Content-Type') or '').lower()
        if 'video' in content_type:
            video = response.content
            break
        try:
            data = response.json()
        except ValueError:
            data = {}
        status = str(data.get('status', '')).upper() if isinstance(data, dict) else ''
        if status == 'COMPLETED':
            url = (data.get('download_url') if isinstance(data, dict) else None) or download_url
            if not url:
                raise ProviderError("The video is ready but Venice gave no download link. Please try again.", "no url")
            file_response = await asyncio.to_thread(_download, url, 180)
            if file_response.status_code != 200:
                raise ProviderError("Couldn't download the finished video from Venice. Please try again.",
                                    f"download {file_response.status_code}")
            video = file_response.content
            break
        if status in ('FAILED', 'ERROR', 'CANCELLED', 'CANCELED'):
            raise ProviderError("Venice couldn't make this video. Try another description or model.", status)
        if on_status and polls % 6 == 0:                     # roughly every 30 seconds
            await on_status(f"🎬 Still working... ({int(waited)} s)")

    if not video:
        raise ProviderError("The video took too long. Venice may still finish it, but I stopped waiting. "
                            "Try again in a bit.", "timeout")

    try:                                                     # tidy up on Venice's side (best effort)
        await _call('/video/complete', {"model": model, "queue_id": queue_id}, timeout=60, max_attempts=1)
    except Exception:
        logger.info("Venice video cleanup call failed (not important)")
    return video, quote
