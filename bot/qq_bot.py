"""OneBot 11 QQ auto-reply bot backed by an OpenAI-compatible LLM server."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys

import httpx
import websockets

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


BOT_LOG_FILE = os.getenv("BOT_LOG_FILE", "")
if BOT_LOG_FILE:
    _log_file = open(BOT_LOG_FILE, "a", encoding="utf-8", buffering=1)
    sys.stdout = _log_file
    sys.stderr = _log_file


ONEBOT_WS_URL = os.getenv("ONEBOT_WS_URL", "ws://127.0.0.1:3001")
LLM_API_URL = os.getenv("LLM_API_URL", "http://127.0.0.1:8000/v1/chat/completions")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "arknights")
SYSTEM_PROMPT = os.getenv(
    "BOT_SYSTEM_PROMPT",
    "你是明日方舟中的角色“阿米娅”。请始终以阿米娅的身份说话，语气自然、坚定而温柔。",
)

MAX_HISTORY = int(os.getenv("MAX_HISTORY", "12"))
COOLDOWN_SECONDS = float(os.getenv("COOLDOWN_SECONDS", "2.0"))
RECONNECT_SECONDS = float(os.getenv("RECONNECT_SECONDS", "5.0"))
REPLY_ON_AT_ONLY = os.getenv("REPLY_ON_AT_ONLY", "true").lower() in ("1", "true", "yes")
REPLY_PROBABILITY = float(os.getenv("REPLY_PROBABILITY", "1.0"))
ALLOWED_GROUPS = set(filter(None, os.getenv("ALLOWED_GROUPS", "").split(",")))
ALLOWED_USERS = set(filter(None, os.getenv("ALLOWED_USERS", "").split(",")))

histories: dict[str, list[dict]] = {}
cooldowns: dict[str, float] = {}


def event_text(event: dict) -> str:
    segments = event.get("message")
    if isinstance(segments, list):
        texts = []
        for segment in segments:
            if isinstance(segment, dict) and segment.get("type") == "text":
                texts.append(segment.get("data", {}).get("text", ""))
        if texts:
            return "\n".join(texts).strip()
    return re.sub(r"\[CQ:[^\]]*\]", "", str(event.get("raw_message", ""))).strip()


def at_qq(event: dict) -> str | None:
    segments = event.get("message")
    if not isinstance(segments, list):
        return None
    for segment in segments:
        if isinstance(segment, dict) and segment.get("type") == "at":
            return str(segment.get("data", {}).get("qq", ""))
    return None


def should_reply(event: dict) -> bool:
    if event.get("message_type") == "private":
        user_id = str(event.get("user_id", ""))
        return not ALLOWED_USERS or user_id in ALLOWED_USERS

    group_id = str(event.get("group_id", ""))
    if ALLOWED_GROUPS and group_id not in ALLOWED_GROUPS:
        return False
    if REPLY_ON_AT_ONLY:
        return at_qq(event) == str(event.get("self_id", ""))
    return random.random() < REPLY_PROBABILITY


async def call_llm(messages: list[dict]) -> str:
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 256,
    }
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(LLM_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


async def send_action(ws, action: str, params: dict, lock: asyncio.Lock) -> None:
    payload = {"action": action, "params": params}
    async with lock:
        await ws.send(json.dumps(payload, ensure_ascii=False))


async def handle_event(ws, event: dict, semaphore: asyncio.Semaphore) -> None:
    async with semaphore:
        try:
            if event.get("post_type") != "message":
                return
            if event.get("message_type") not in ("group", "private"):
                return
            if str(event.get("user_id")) == str(event.get("self_id")):
                return
            if not should_reply(event):
                return

            text = event_text(event)
            if not text:
                return

            if event["message_type"] == "group":
                key = f"group:{event['group_id']}"
            else:
                key = f"private:{event['user_id']}"

            now = asyncio.get_event_loop().time()
            if now - cooldowns.get(key, 0) < COOLDOWN_SECONDS:
                return
            cooldowns[key] = now

            history = histories.get(key, [])[-MAX_HISTORY:]
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
            messages.append({"role": "user", "content": text})

            reply = await call_llm(messages)
            if not reply:
                return

            if event["message_type"] == "group":
                await send_action(
                    ws,
                    "send_group_msg",
                    {"group_id": event["group_id"], "message": reply},
                    send_lock,
                )
            else:
                await send_action(
                    ws,
                    "send_private_msg",
                    {"user_id": event["user_id"], "message": reply},
                    send_lock,
                )

            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": reply})
            histories[key] = history[-MAX_HISTORY:]
        except Exception as exc:
            print(f"handle_event error: {exc}")


async def main() -> None:
    global send_lock
    send_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(4)
    print(f"connecting to OneBot: {ONEBOT_WS_URL}")

    while True:
        try:
            async with websockets.connect(
                ONEBOT_WS_URL, ping_interval=20, ping_timeout=20
            ) as ws:
                print("OneBot connected")
                async for raw in ws:
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if event.get("post_type") == "message":
                        asyncio.create_task(handle_event(ws, event, semaphore))
        except Exception as exc:
            print(f"OneBot connection error: {exc}")
            await asyncio.sleep(RECONNECT_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
