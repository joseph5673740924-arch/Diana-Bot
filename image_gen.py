"""
image_gen.py — 黛安娜「動作插圖」自動生成模組
--------------------------------------------------
跟 bot.py / diana_evolution.py 一樣走「金鑰互相分離」的原則，使用專屬的
config/google_api_key_image.txt，呼叫 Gemini 圖片生成 API
(gemini-3.1-flash-image，俗稱 Nano Banana)，並帶入角色參考圖
(img/reference/ 資料夾內的三視圖 / 範例圖) 來維持角色一致性，
生成新的「動作情境插圖」。

使用方式（在 bot.py 裡）：
    from image_gen import generate_action_image, ImageGenError

    try:
        image_bytes, mime_type = await generate_action_image("歪頭想了想")
    except ImageGenError as e:
        ...

這支模組只負責「呼叫 API + 回傳圖片 bytes」，存檔、寫入 action_images.json
對照表等後續動作交給 bot.py 的指令處理，方便你之後想改存檔規則或加審核流程。
"""
import os
import base64
import hashlib
import mimetypes
import time
import json

import aiohttp

CONFIG_DIR = "config"
IMG_DIR = "img"
REFERENCE_DIR = os.path.join(IMG_DIR, "reference")
IMAGE_API_KEY_FILE = os.path.join(CONFIG_DIR, "google_api_key_image.txt")

# GUI 只覆寫連線設定，不改變原本圖片生成流程。
def _load_gui_image_config():
    try:
        with open(os.path.join(CONFIG_DIR, "gui_config.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

_gui_cfg = _load_gui_image_config()
IMAGE_MODEL = _gui_cfg.get("image_model", "gemini-3.1-flash-image")
GEMINI_ENDPOINT = _gui_cfg.get(
    "image_endpoint",
    f"https://generativelanguage.googleapis.com/v1beta/models/{IMAGE_MODEL}:generateContent"
)

# 角色基本設定，讓每次生圖都盡量維持一致的人設，內容可依實際角色設定微調。
CHARACTER_STYLE_PROMPT = (
    "角色是「黛安娜」，一位金髮綠眼、有貓耳貓尾的少女機器人角色，"
    "穿著藍色迷彩外套。請完全比照附帶的參考圖片的動漫上色風格與角色設計來畫，"
    "五官、髮型、貓耳貓尾、服裝、身形比例都必須跟參考圖保持一致，不能變成別的角色或別的畫風。"
)

CONSTRAINT_PROMPT = (
    "構圖限制：\n"
    "1. 純白色背景（或完全透明背景），畫面中不可以出現任何文字、浮水印、對話框、Logo 或商標。\n"
    "2. 畫面中只能有黛安娜一個角色，半身或全身皆可。\n"
    "3. 姿勢、表情、動作需要符合下面描述的『動作情境』。\n"
    "4. 圖片維持直向或方形構圖，方便之後直接貼在 Discord 訊息裡顯示。"
)


class ImageGenError(Exception):
    """圖片生成流程中任何可預期的失敗都用這個例外包起來，方便上層統一處理、回報給主人。"""
    pass


def get_image_api_key():
    if not os.path.exists(IMAGE_API_KEY_FILE):
        raise ImageGenError(
            f"找不到圖片生成專用金鑰 '{IMAGE_API_KEY_FILE}'，"
            f"請建立該檔案並貼入專門用於生圖的 Google API Key"
            f"（建議跟聊天 google_api_key_chat.txt、演化 google_api_key_evolution.txt 分開申請，避免搶額度）。"
        )
    with open(IMAGE_API_KEY_FILE, "r", encoding="utf-8") as f:
        key = f.readline().strip()
    if not key:
        raise ImageGenError(f"'{IMAGE_API_KEY_FILE}' 內容是空的，請貼入有效的 API Key。")
    return key


def _load_reference_images():
    """讀取 img/reference/ 資料夾內所有圖片，轉成 Gemini API 需要的 inline_data 格式。"""
    if not os.path.isdir(REFERENCE_DIR):
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        raise ImageGenError(
            f"'{REFERENCE_DIR}' 資料夾是空的（剛幫你建好了），"
            f"請先把角色三視圖 / 幾張範例插圖放進這個資料夾（建議 1~4 張，含正面、側面、常見表情），"
            f"這些圖會在每次生圖時一起送給 AI 當作『畫這個角色』的參考依據。"
        )
    parts = []
    for fname in sorted(os.listdir(REFERENCE_DIR)):
        fpath = os.path.join(REFERENCE_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        mime_type, _ = mimetypes.guess_type(fpath)
        if not mime_type or not mime_type.startswith("image/"):
            continue
        with open(fpath, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")
        parts.append({"inline_data": {"mime_type": mime_type, "data": b64_data}})
    if not parts:
        raise ImageGenError(
            f"'{REFERENCE_DIR}' 資料夾裡沒有找到任何可用的圖片檔，至少需要放 1 張角色參考圖。"
        )
    return parts


async def generate_action_image(action_description: str, extra_notes: str = ""):
    """
    呼叫 Gemini 圖片生成 API，依角色參考圖 + 動作描述生成一張新插圖。

    參數：
        action_description: 動作描述文字，例如「歪頭想了想」。
        extra_notes: 額外補充的畫面要求（可留空），例如「表情要驚訝一點」。

    回傳：
        (圖片 bytes, mime_type)

    失敗時丟出 ImageGenError，內含可以直接回報給使用者看的中文錯誤訊息。
    """
    api_key = get_image_api_key()
    reference_parts = _load_reference_images()

    prompt_text = (
        f"{CHARACTER_STYLE_PROMPT}\n\n"
        f"請生成的動作情境：「{action_description}」\n"
        + (f"補充要求：{extra_notes}\n\n" if extra_notes else "\n")
        + CONSTRAINT_PROMPT
    )

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": reference_parts + [{"text": prompt_text}]
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE"]
        }
    }
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GEMINI_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=90)
            ) as resp:
                data = await resp.json()
                if resp.status != 200:
                    err_msg = data.get("error", {}).get("message", str(data))
                    raise ImageGenError(f"圖片生成 API 呼叫失敗 (HTTP {resp.status})：{err_msg}")
    except aiohttp.ClientError as e:
        raise ImageGenError(f"連線圖片生成 API 時發生網路錯誤：{e}")

    try:
        candidates = data["candidates"]
        parts = candidates[0]["content"]["parts"]
        for part in parts:
            inline_data = part.get("inlineData") or part.get("inline_data")
            if inline_data and inline_data.get("data"):
                image_bytes = base64.b64decode(inline_data["data"])
                mime_type = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"
                return image_bytes, mime_type
    except (KeyError, IndexError) as e:
        raise ImageGenError(f"回應格式不如預期，無法解析出圖片內容：{e}\n原始回應：{data}")

    raise ImageGenError(f"API 回應中沒有找到圖片資料，可能是內容被安全機制擋掉了。原始回應：{data}")


def suggest_filename(action_description: str, ext: str = ".png") -> str:
    """
    幫自動生成的圖片取一個安全的英數檔名：action_auto_{時間戳記}_{內容雜湊}.ext
    （中文動作描述直接拿來當檔名容易在不同作業系統上出問題，所以不用中文原文當檔名，
    但會把原文完整存進 action_images.json 的 "name" 欄位方便你之後辨識。）
    """
    digest = hashlib.sha1(action_description.encode("utf-8")).hexdigest()[:8]
    ts = int(time.time())
    return f"action_auto_{ts}_{digest}{ext}"
