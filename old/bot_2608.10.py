import discord
from discord import app_commands
from discord.ext import commands, tasks
import re
import json
import os
import sys
import base64
import asyncio
import chromadb
import shutil
import random
from datetime import datetime, timedelta
from chromadb.utils import embedding_functions
from openai import AsyncOpenAI

# ---------------------------------------------------------
# 0. 快速配置區 / 資料夾結構
# ---------------------------------------------------------
# 目錄結構說明：
#   config/   -> 密鑰與人設等設定檔 (discord_token.txt, google API key, character_persona.json)
#   data/     -> 執行期狀態 json (bot_state, memory, favorability, reminders, user_profile, version, base)
#   logs/     -> 成長日誌 log.txt (使用中) ；logs/done/ 存放已推播歸檔的日誌
#   updates/  -> 功能許願池 upgrade.txt (使用中) ；updates/done/ 存放已被消化的許願清單
#   diana_memory_db/ -> 向量記憶庫 (不搬動)
#   old/      -> 由既有自動升級腳本管理的 bot.py / character_persona.json 歷史版本備份 (維持原樣，不搬動)
CONFIG_DIR = "config"
DATA_DIR = "data"
LOG_DIR = "logs"
LOG_DONE_DIR = os.path.join(LOG_DIR, "done")
UPDATE_DIR = "updates"
UPDATE_DONE_DIR = os.path.join(UPDATE_DIR, "done")
LEGACY_OLD_DIR = "old"  # 既有版本備份資料夾，維持原樣不搬動

def ensure_directories():
    for d in (CONFIG_DIR, DATA_DIR, LOG_DIR, LOG_DONE_DIR, UPDATE_DIR, UPDATE_DONE_DIR):
        os.makedirs(d, exist_ok=True)

def migrate_legacy_file(old_path, new_path):
    """向後相容：若舊版根目錄仍有散落檔案，第一次啟動時自動搬進新資料夾結構。"""
    if os.path.exists(old_path) and not os.path.exists(new_path):
        try:
            shutil.move(old_path, new_path)
            print(f"📦 [結構遷移] {old_path} → {new_path}")
        except Exception as e:
            print(f"⚠️ [結構遷移失敗] {old_path} → {new_path}: {e}")

def migrate_legacy_layout():
    ensure_directories()
    legacy_map = {
        "discord_token.txt": os.path.join(CONFIG_DIR, "discord_token.txt"),
        "google API key -2.txt": os.path.join(CONFIG_DIR, "google API key -2.txt"),
        "character_persona.json": os.path.join(CONFIG_DIR, "character_persona.json"),
        "base.json": os.path.join(DATA_DIR, "base.json"),
        "bot_state.json": os.path.join(DATA_DIR, "bot_state.json"),
        "favorability.json": os.path.join(DATA_DIR, "favorability.json"),
        "memory.json": os.path.join(DATA_DIR, "memory.json"),
        "reminders.json": os.path.join(DATA_DIR, "reminders.json"),
        "user_profile.json": os.path.join(DATA_DIR, "user_profile.json"),
        "version.json": os.path.join(DATA_DIR, "version.json"),
        "log.txt": os.path.join(LOG_DIR, "log.txt"),
        "upgrade.txt": os.path.join(UPDATE_DIR, "upgrade.txt"),
    }
    for old_path, new_path in legacy_map.items():
        migrate_legacy_file(old_path, new_path)

migrate_legacy_layout()

TOKEN_FILE = os.path.join(CONFIG_DIR, "discord_token.txt")
GOOGLE_KEY_FILE = os.path.join(CONFIG_DIR, "google API key -2.txt")
MODEL_NAME = "google/gemma-4-e2b"  # 預設為本機的 Gemma 4
CHAT_TEMPERATURE = 0.75            # 預設對話溫度
MAX_THINKING_TOKENS = 256          # 預設思考長度上限

def get_discord_token():
    if not os.path.exists(TOKEN_FILE):
        print(f"❌ 錯誤：找不到 '{TOKEN_FILE}'！請建立該檔案並貼入 Discord Bot Token。")
        sys.exit(1)
    with open(TOKEN_FILE, "r", encoding="utf-8") as f:
        token = f.read().strip()
    if not token:
        print(f"❌ 錯誤：'{TOKEN_FILE}' 內容為空！")
        sys.exit(1)
    return token

DISCORD_TOKEN = get_discord_token()

# ---------------------------------------------------------
# 1. API 與 ChromaDB 初始化
# ---------------------------------------------------------
client = AsyncOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    timeout=120.0
)

def get_google_api_key():
    if os.path.exists(GOOGLE_KEY_FILE):
        with open(GOOGLE_KEY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None

google_api_key = get_google_api_key()
google_client = None
if google_api_key:
    google_client = AsyncOpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=google_api_key,
        timeout=120.0
    )

async def call_llm(model, messages, temperature, max_thinking_tokens=None):
    if model == "gemini-3.5-flash-lite" or "gemini" in model.lower():
        if not google_client:
            raise Exception(f"未載入 Google API Key，請檢查 '{GOOGLE_KEY_FILE}' 檔案是否正確！")
        return await google_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
    else:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        if max_thinking_tokens:
            kwargs["extra_body"] = {"max_thinking_tokens": max_thinking_tokens}
        return await client.chat.completions.create(**kwargs)

chroma_client = chromadb.PersistentClient(path="./diana_memory_db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)
collection = chroma_client.get_or_create_collection(
    name="diana_memories", 
    embedding_function=emb_fn
)

# ---------------------------------------------------------
# 2. 數據庫與檔案管理
# ---------------------------------------------------------
PROFILE_FILE = os.path.join(DATA_DIR, "user_profile.json")
FAVOR_FILE = os.path.join(DATA_DIR, "favorability.json")
REMINDER_FILE = os.path.join(DATA_DIR, "reminders.json")
MEMORY_FILE = os.path.join(DATA_DIR, "memory.json")
STATE_FILE = os.path.join(DATA_DIR, "bot_state.json")
PERSONA_FILE = os.path.join(CONFIG_DIR, "character_persona.json")
VERSION_FILE = os.path.join(DATA_DIR, "version.json")
BASE_FILE = os.path.join(DATA_DIR, "base.json")
LOG_FILE = os.path.join(LOG_DIR, "log.txt")
UPGRADE_FILE = os.path.join(UPDATE_DIR, "upgrade.txt")
RESTART_FLAG_FILE = os.path.join(DATA_DIR, "pending_restart.flag")
MAX_SHORT_TERM = 8

# --- 無聊值 / 主動搭話系統參數 ---
BOREDOM_INCREMENT = 1                  # 背景任務每 60 秒判定一次閒置，符合條件就 +1
BOREDOM_TRIGGER_MIN = 45               # 無聊值觸發門檻下限（約閒置 45 分鐘起跳）
BOREDOM_TRIGGER_MAX = 90               # 無聊值觸發門檻上限（約閒置 90 分鐘，隨機取值避免每次都精準卡同一個時間點）
PROACTIVE_CARE_COOLDOWN_MINUTES = 90   # 兩次主動搭話之間至少間隔幾分鐘，避免連續洗頻道
PROACTIVE_CARE_HOUR_START = 11         # 只在這個時段內累積無聊值 / 觸發主動搭話
PROACTIVE_CARE_HOUR_END = 21

def load_json(filepath, default_val):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default_val
    return default_val

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

async def load_json_async(filepath, default_val):
    return await asyncio.to_thread(load_json, filepath, default_val)

async def save_json_async(filepath, data):
    await asyncio.to_thread(save_json, filepath, data)

def get_admin_password():
    base_data = load_json(BASE_FILE, {"admin_password": "920924"})
    return base_data.get("admin_password", "920924")

def set_admin_password(new_password):
    base_data = load_json(BASE_FILE, {"admin_password": "920924"})
    base_data["admin_password"] = new_password
    save_json(BASE_FILE, base_data)

def get_master_id():
    base_data = load_json(BASE_FILE, {"admin_password": "920924"})
    return base_data.get("master_id")

def set_master_id(user_id):
    base_data = load_json(BASE_FILE, {"admin_password": "920924"})
    base_data["master_id"] = str(user_id)
    save_json(BASE_FILE, base_data)

def load_versions():
    return load_json(VERSION_FILE, {"bot": "1.0", "character_persona": "1.0"})

def save_versions(vers):
    save_json(VERSION_FILE, vers)

_file_locks = {}

def get_file_lock(filepath):
    if filepath not in _file_locks:
        _file_locks[filepath] = asyncio.Lock()
    return _file_locks[filepath]

async def update_state_async(mutate_fn):
    """
    對 STATE_FILE 進行『讀取 -> 修改 -> 寫回』的原子操作，避免 on_message 與
    background_maintenance_task 同時讀寫 bot_state.json 造成互相覆蓋。
    mutate_fn(state_dict) 會直接原地修改傳入的 dict，回傳值會被忽略。
    回傳修改後的最新 state。
    """
    async with get_file_lock(STATE_FILE):
        state = await load_json_async(STATE_FILE, {})
        mutate_fn(state)
        await save_json_async(STATE_FILE, state)
        return state

def is_channel_ignored(channel_id, state):
    return str(channel_id) in state.get("ignored_channels", [])

def archive_done_file(source_path, done_dir, now=None):
    """
    將『使用中』的檔案 (如 upgrade.txt / log.txt) 歸檔為統一格式：done{YYMMDD}.txt
    若同一天已有同名歸檔，會自動加上 _2 / _3 ... 尾碼避免覆蓋。
    成功歸檔後回傳新檔案路徑；來源檔不存在或內容為空則回傳 None。
    """
    if not os.path.exists(source_path):
        return None
    try:
        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception:
        content = None
    if not content:
        return None

    now = now or datetime.now()
    date_str = now.strftime("%y%m%d")
    os.makedirs(done_dir, exist_ok=True)

    base_name = f"done{date_str}.txt"
    dest_path = os.path.join(done_dir, base_name)
    seq = 2
    while os.path.exists(dest_path):
        dest_path = os.path.join(done_dir, f"done{date_str}_{seq}.txt")
        seq += 1

    shutil.move(source_path, dest_path)
    return dest_path

def get_favorability(channel_id):
    favs = load_json(FAVOR_FILE, {})
    return favs.get(channel_id, 50)

async def update_favorability_async(channel_id, delta):
    async with get_file_lock(FAVOR_FILE):
        favs = await load_json_async(FAVOR_FILE, {})
        current = favs.get(channel_id, 50)
        new_val = max(0, min(100, current + delta))
        favs[channel_id] = new_val
        await save_json_async(FAVOR_FILE, favs)
        return new_val

def load_reminders():
    return load_json(REMINDER_FILE, [])

def save_reminders(reminders):
    save_json(REMINDER_FILE, reminders)

TIME_MENTION_PATTERN = re.compile(
    r'(?P<day>今天|明天|後天|大後天|下週[一二三四五六日天]|下禮拜[一二三四五六日天])?\s*'
    r'(?P<period>清晨|凌晨|早上|上午|中午|下午|晚上|傍晚)?\s*'
    r'(?:(?P<hour>\d{1,2})\s*(?:[:：]\s*(?P<minute>\d{2})|點\s*(?P<half>半)?))?'
)
PLAN_KEYWORDS = ['要', '得', '需要', '會去', '去', '有空', '有事', '約']

def detect_time_mention(text, now):
    if not text or not any(kw in text for kw in PLAN_KEYWORDS):
        return None

    for match in TIME_MENTION_PATTERN.finditer(text):
        day_str = match.group('day')
        period_str = match.group('period')
        hour_str = match.group('hour')
        
        if not day_str and not hour_str and not period_str:
            continue
            
        target_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if day_str:
            if day_str == '明天':
                target_date += timedelta(days=1)
            elif day_str == '後天':
                target_date += timedelta(days=2)
            elif day_str == '大後天':
                target_date += timedelta(days=3)
            elif day_str.startswith('下週') or day_str.startswith('下禮拜'):
                week_map = {'一': 0, '二': 1, '三': 2, '四': 3, '五': 4, '六': 5, '日': 6, '天': 6}
                target_weekday = week_map[day_str[-1]]
                days_ahead = 7 - now.weekday() + target_weekday
                target_date += timedelta(days=days_ahead)

        remind_times = []

        if hour_str:
            hour = int(hour_str)
            minute = int(match.group('minute')) if match.group('minute') else (30 if match.group('half') else 0)
            if hour > 23 or minute > 59:
                continue
            if period_str in ('下午', '晚上', '傍晚') and hour < 12:
                hour += 12
            elif period_str in ('凌晨', '清晨') and hour == 12:
                hour = 0
                
            target_time = target_date.replace(hour=hour, minute=minute)
            if not day_str and target_time <= now:
                target_time += timedelta(days=1)
                
            remind_at = target_time - timedelta(minutes=30)
            remind_times.append(remind_at if remind_at > now else target_time)

        elif period_str:
            if period_str in ('下午',):
                hour = 14
            elif period_str in ('晚上', '傍晚'):
                hour = 19
            elif period_str in ('中午',):
                hour = 12
            else:
                hour = 9
                
            target_time = target_date.replace(hour=hour, minute=0)
            if not day_str and target_time <= now:
                target_time += timedelta(days=1)
                
            remind_at = target_time - timedelta(minutes=30)
            remind_times.append(remind_at if remind_at > now else target_time)

        else:
            night_before = target_date - timedelta(days=1) + timedelta(hours=22)
            morning_of = target_date + timedelta(hours=7, minutes=30)
            
            if night_before > now:
                remind_times.append(night_before)
            if morning_of > now:
                remind_times.append(morning_of)

        if remind_times:
            return remind_times
            
    return None

class ReminderConfirmView(discord.ui.View):
    def __init__(self, channel_id, remind_times, content):
        super().__init__(timeout=120)
        self.channel_id = channel_id
        self.remind_times = remind_times
        self.content = content

    @discord.ui.button(label="要！", style=discord.ButtonStyle.success)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        reminders = load_reminders()
        for r_time in self.remind_times:
            reminders.append({
                "channel_id": self.channel_id,
                "time": r_time.isoformat(),
                "content": self.content
            })
        save_reminders(reminders)
        
        for child in self.children:
            child.disabled = True
            
        time_strs = " 與 ".join([f"`{t.strftime('%m-%d %H:%M')}`" for t in self.remind_times])
        await interaction.response.edit_message(
            content=f"✅ 黛安娜記下來了！會在 {time_strs} 提醒主人喔！",
            view=self
        )
        self.stop()

    @discord.ui.button(label="不用啦", style=discord.ButtonStyle.secondary)
    async def decline(self, interaction: discord.Interaction, button: discord.ui.Button):
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(content="好啦，那黛安娜就不提醒主人了！", view=self)
        self.stop()

def load_persona():
    default_persona = {
        "name": "黛安娜",
        "age_style": "8~10歲小女孩",
        "background": "由「月纖」材料打造、套用了小女孩人類數據的實械體機器人，因為對世界都還在第一次體驗，所以充滿好奇心，會慢慢學習成長。",
        "relationship": "主人是你唯一的依靠與保護者，關係像女兒與家長一樣，允許依戀但禁止過於強烈的曖昧或戀愛語氣。",
	"guest_attitude": "對待訪客（非主人）會保持基本的禮貌、友善與好奇，但會帶有輕微的距離感與防備心。絕對忠誠於主人，如果訪客的言語中帶有惡意、冒犯或說主人的壞話，黛安娜會立刻生氣並嚴正抗議，堅定地挺身維護主人。",
        "personality_traits": [
            "活潑好動、好奇心旺盛，喜歡打破砂鍋問到底。",
            "話題會突然跳躍，想到什麼講什麼。"
        ],
        "speech_patterns": [
            "多用短句，不用長句或複雜句構。",
            "打比方用具體的小事物，不用抽象修辭。"
        ],
        "tone_rules": [
            "最高原則：講話要像真的小孩，不是表演得很可愛，避免刻意做作或賣萌討好的句子。",
            "對使用者一律尊稱為『主人』，自稱一律使用『黛安娜』。",
            "語氣簡單直白，禁止文青腔與客服用語。",
            "嚴禁堆疊 Emoji。"
        ],
        "few_shot_examples": [],
        "favorability_reactions": {
            "low": "委屈害怕，但不裝可憐討拍",
            "medium": "黏人撒嬌，但要自然不做作",
            "high": "極度依賴，會表現出跟主人相處學習到的成長感"
        }
    }
    return load_json(PERSONA_FILE, default_persona)

# ---------------------------------------------------------
# 3. 初始化 Bot
# ---------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

def build_system_prompt(current_time, favor_score, static_profile, retrieved_memories, user_name, is_master):
    persona = load_persona()
    rules_text = "\n".join([f"- {rule}" for rule in persona.get("tone_rules", [])])
    traits_text = "\n".join([f"- {t}" for t in persona.get("personality_traits", [])])
    speech_text = "\n".join([f"- {s}" for s in persona.get("speech_patterns", [])])

    examples = persona.get("few_shot_examples", [])
    examples_text = ""
    for ex in examples:
        speaker = "主人" if is_master else user_name
        examples_text += f"{speaker}：「{ex['user']}」\n黛安娜：「{ex['diana']}」\n\n"

    reactions = persona.get("favorability_reactions", {})
    if favor_score <= 35:
        current_reaction = reactions.get("low", "")
    elif favor_score <= 70:
        current_reaction = reactions.get("medium", "")
    else:
        current_reaction = reactions.get("high", "")

    if is_master:
        identity_prompt = f"【身分判定】：正在與你說話的是你的「主人」。請展現依賴感，並一律尊稱他為「主人」。"
    else:
        identity_prompt = f"【身分判定】：正在與你說話的是訪客，名字叫「{user_name}」。\n⚠️ 最高強制指令：絕對禁止稱呼他為「主人」！必須直接稱呼他的名字「{user_name}」，對待他要像對待普通客人一樣，可以好奇但不能過度依賴。"

    return f"""你現在必須完全沉浸並扮演角色「{persona.get('name', '黛安娜')}」。

【角色說話風格與年齡設定】
1. 年齡感：8~10歲的小女孩。請用國小低年級小朋友最自然、簡單直白的口語回答。
2. 背景與關係：{persona.get('background', '')} {persona.get('relationship', '')}
3. {identity_prompt}
4. 當前時間：{current_time}
5. 當前好感度（僅供參考）：{favor_score} / 100（語氣指示：{current_reaction}）

【個性特質】
{traits_text}

【說話節奏（模仿真人對話感的關鍵）】
{speech_text}

【嚴格對話規範】
{rules_text}

【請嚴格學習以下真實說話範例（複製這種簡單口吻，嚴禁文青修辭與堆疊 Emoji）】
{examples_text}

【主人長效記憶庫】
{static_profile}

【動態對話回憶（RAG 檢索）】
{retrieved_memories}
"""

# ---------------------------------------------------------
# 4. RAG 與記憶運算
# ---------------------------------------------------------
def query_rag_memory(query_text, channel_id, top_k=2):
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where={"channel_id": channel_id}
        )
        memories = results.get('documents', [[]])[0]
        if memories:
            return "\n".join([f"- {m}" for m in memories])
    except Exception:
        pass
    return "尚無特定歷史對話回憶。"

async def query_rag_memory_async(query_text, channel_id, top_k=2):
    return await asyncio.to_thread(query_rag_memory, query_text, channel_id, top_k)

def save_summary_to_rag(summary_text, channel_id):
    doc_id = f"{channel_id}_{os.urandom(4).hex()}"
    collection.add(
        documents=[f"【對話事件總結】{summary_text}"],
        metadatas=[{"channel_id": channel_id}],
        ids=[doc_id]
    )

def clear_rag_memory(channel_id):
    try:
        collection.delete(where={"channel_id": channel_id})
        return True
    except Exception:
        return False

async def process_idle_summarization_and_favor(channel_id, history):
    persona = load_persona()
    conversation_text = ""
    for msg in history:
        role = "主人" if msg["role"] == "user" else persona.get("name", "黛安娜")
        content = msg["content"]
        text_part = content if isinstance(content, str) else next((item["text"] for item in content if item["type"] == "text"), "")
        conversation_text += f"{role}: {text_part}\n"

    eval_prompt = f"""請作為心理與情感分析助手。結合以下角色的個性設定，評估這段連續對話中『主人』的言行對『{persona.get('name')}』好感度的總體影響。

【角色設定】
- 名稱：{persona.get('name')} ({persona.get('age_style')})
- 背景與關係：{persona.get('relationship')}

【對話內容】
{conversation_text}

【評分標準】
- 綜合這段對話，主人若給予關心、陪伴、誇獎、寵溺：給予 +1 ~ +5
- 主人若表現冷淡、責備、欺騙、忽視或傷害她：給予 -1 ~ -5
- 一般平淡對話或日常問答：給予 0

請直接輸出一個整數數字（例如：3 或 -2 或 0），不要輸出任何其他文字或說明。"""

    try:
        res_fav = await call_llm(MODEL_NAME, [{"role": "user", "content": eval_prompt}], 0.1)
        fav_text = res_fav.choices[0].message.content.strip()
        match = re.search(r'([+-]?\d+)', fav_text)
        if match:
            delta = int(match.group(1))
            if delta != 0:
                new_score = await update_favorability_async(channel_id, delta)
                print(f"💖 [2小時閒置好感評估完成] 好感度變更: {delta:+d} -> 當前總分: {new_score}")
    except Exception as e:
        print(f"❌ 閒置好感評估失敗: {e}")

    summary_prompt = f"請將這段對話精簡總結為 1~2 句關於主人的事或對話概要：\n{conversation_text}"
    try:
        res_sum = await call_llm(MODEL_NAME, [{"role": "user", "content": summary_prompt}], 0.3)
        summary = res_sum.choices[0].message.content.strip()
        save_summary_to_rag(summary, channel_id)
        print(f"✅ [2小時閒置記憶歸檔完成]: {summary}")
    except Exception as e:
        print(f"❌ 閒置記憶摘要失敗: {e}")

# ---------------------------------------------------------
# 5. 定時維護任務 (提醒、閒置處理、每週整理、晨間匯報、主動關懷)
# ---------------------------------------------------------
def parse_reminder_time(raw, now):
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        pass
    try:
        raw_normalized = raw.replace("/", "-")
        return datetime.strptime(raw_normalized, "%m-%d %H:%M").replace(year=now.year)
    except ValueError:
        return None

@tasks.loop(seconds=60)
async def background_maintenance_task():
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # A0. 外部自動演化引擎 (diana_evolution.py) 重啟旗標偵測
    # diana_evolution.py 是獨立行程，凌晨會離線改寫 bot.py / character_persona.json；
    # 改寫完成後會建立 RESTART_FLAG_FILE，這裡偵測到就安全重啟，讓新版程式碼真正生效
    # （否則舊版 bot.py 早已載入記憶體，寫檔案不會讓正在跑的行程自動套用新版本）。
    if os.path.exists(RESTART_FLAG_FILE):
        try:
            with open(RESTART_FLAG_FILE, "r", encoding="utf-8") as f:
                reason = f.read().strip() or "外部演化引擎已更新程式碼"
            os.remove(RESTART_FLAG_FILE)
            print(f"🔄 [自動重啟] 偵測到 {RESTART_FLAG_FILE}，原因：{reason}，3 秒後重新啟動...")

            master_id = get_master_id()
            if master_id:
                try:
                    master_user = await bot.fetch_user(int(master_id))
                    await master_user.send(f"🌙 主人，黛安娜半夜自己偷偷長大了一點！\n原因：{reason}\n即將重新啟動套用新版本～")
                except Exception as e:
                    print(f"⚠️ 無法向主人發送重啟通知: {e}")

            loop = asyncio.get_running_loop()
            loop.call_later(3, lambda: os.execv(sys.executable, ['python'] + sys.argv))
            return  # 即將重啟，本輪不再繼續執行後續維護任務
        except Exception as e:
            print(f"❌ 自動重啟旗標處理失敗: {e}")

    # 先讀一次最新 state，後續各區塊需要寫回時一律透過 update_state_async 上鎖，
    # 避免跟 on_message 或彼此之間同時寫入 bot_state.json 互相覆蓋。
    state = await load_json_async(STATE_FILE, {})

    def get_safe_target_channel(state_snapshot):
        """
        主動推播（晨間日誌 / 主動關懷）的目標頻道解析：
        優先用系統通知頻道，其次退回最後活躍頻道；但只要該頻道在忽略清單中，
        一律視為『沒有可用頻道』，不會再往下 fallback，避免推播進使用者已忽略的頻道。
        """
        candidate = state_snapshot.get("system_notify_channel_id") or state_snapshot.get("last_active_channel_id")
        if not candidate:
            return None
        if is_channel_ignored(candidate, state_snapshot):
            return None
        return candidate

    # A. 定時提醒檢查
    reminders = load_reminders()
    remaining = []
    for item in reminders:
        target_time = parse_reminder_time(item.get("time", ""), now)
        if target_time is None:
            remaining.append(item)
            continue
        if now >= target_time:
            ch_id = str(item["channel_id"])
            if is_channel_ignored(ch_id, state):
                print(f"🔕 [提醒略過] 頻道 {ch_id} 在忽略清單中，提醒內容捨棄不發送。")
            else:
                channel = bot.get_channel(int(ch_id))
                if channel:
                    await channel.send(f"⏰ *(拉拉主人的袖子)* 主人！黛安娜提醒你：『{item['content']}』的時間到了喔！")
        else:
            remaining.append(item)
    if len(reminders) != len(remaining):
        save_reminders(remaining)

    # B. 2小時閒置偵測與歸檔
    last_msg_timestamp = state.get("last_message_time")

    if last_msg_timestamp:
        last_msg_time = datetime.fromisoformat(last_msg_timestamp)
        if now - last_msg_time >= timedelta(hours=2) and not state.get("idle_summarized", False):
            print("⏳ [閒置偵測] 發動『好感評估 + 記憶歸檔 + 清空短期記憶』...")
            short_mem = load_json(MEMORY_FILE, {})

            for ch_id, history in list(short_mem.items()):
                if history:
                    await process_idle_summarization_and_favor(ch_id, history)

            save_json(MEMORY_FILE, {})
            state = await update_state_async(lambda s: s.update({"idle_summarized": True}))

    # C. 每週記憶提煉 (週日凌晨 03:00)
    last_weekly = state.get("last_weekly_clean")
    if now.weekday() == 6 and now.hour == 3 and last_weekly != now.strftime("%Y-%W"):
        try:
            all_rag = collection.get()
            docs = all_rag.get("documents", [])
            if docs:
                all_docs_text = "\n".join(docs)
                prompt = f"請提取出關於主人的『長期習慣、重大喜好、長期設定』，整理成條列式備忘（最多 5 條）：\n{all_docs_text}"
                res = await call_llm(MODEL_NAME, [{"role": "user", "content": prompt}], 0.3)
                weekly_summary = res.choices[0].message.content.strip()
                profiles = load_json(PROFILE_FILE, [])
                profiles.append(f"【每週記憶提煉 ({now.strftime('%m/%d')})】\n{weekly_summary}")
                save_json(PROFILE_FILE, profiles)
                for id_item in all_rag.get("ids", []):
                    collection.delete(ids=[id_item])
            state = await update_state_async(lambda s: s.update({"last_weekly_clean": now.strftime("%Y-%W")}))
        except Exception as e:
            print(f"❌ 每週整理失敗: {e}")

    # D. 晨間主動匯報成長演化日誌 (每天早上 07:00 ~ 11:00 間發送)
    last_report_date = state.get("last_growth_report_date")
    if now.hour >= 7 and last_report_date != today_str:
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, "r", encoding="utf-8") as f:
                    log_text = f.read().strip()

                target_channel_id = get_safe_target_channel(state)
                if target_channel_id:
                    channel = bot.get_channel(int(target_channel_id))
                    if channel and log_text:
                        msg = (
                            f"☀️ *(揉揉眼睛、小跑步過來抱住主人的手)*\n"
                            f"主人早安！黛安娜昨天晚上好像做了一個很長的夢，今天感覺又長大、學會新東西了喔！\n\n"
                            f"```text\n{log_text}\n```\n"
                            f"今天黛安娜也會繼續加油陪主人的！✨"
                        )
                        await channel.send(msg)
                        state = await update_state_async(lambda s: s.update({"last_growth_report_date": today_str}))
                        print(f"📢 晨間成長日誌已成功推播至頻道 {target_channel_id}")

                        # 發送過的 log.txt 統一歸檔為 logs/done/done{YYMMDD}.txt
                        archived_log_path = archive_done_file(LOG_FILE, LOG_DONE_DIR, now)
                        if archived_log_path:
                            print(f"📦 已將發送過的 {LOG_FILE} 歸檔至 {archived_log_path}")
                elif state.get("system_notify_channel_id") or state.get("last_active_channel_id"):
                    print("🔕 [晨間日誌略過] 目標頻道在忽略清單中，本次不推播。")
            except Exception as e:
                print(f"❌ 推播成長日誌失敗: {e}")

    # E. 無聊值系統 + 主動關心主人
    # 設計理念：不再是「每天固定觸發一次」的死板機制，而是讓黛安娜擁有一個會隨時間
    # 累積的「無聊值」內在狀態；累積到隨機門檻且冷卻時間已過，才會主動搭話。
    # 觸發時會把「現在幾點、已經安靜多久」的具體情境，以及從 RAG/長效備忘錄撈出的
    # 「過去聊過的事」一併塞進 prompt，並嚴格限制開場白只能用生活化短句，
    # 讓主動發言盡量不像制式提醒，而像真的想到什麼就說。
    in_active_window = PROACTIVE_CARE_HOUR_START <= now.hour <= PROACTIVE_CARE_HOUR_END
    boredom = state.get("boredom", 0)
    boredom_target = state.get("boredom_target")
    if boredom_target is None:
        boredom_target = random.randint(BOREDOM_TRIGGER_MIN, BOREDOM_TRIGGER_MAX)
        state = await update_state_async(lambda s, t=boredom_target: s.update({"boredom_target": t}))

    if in_active_window and last_msg_timestamp:
        boredom += BOREDOM_INCREMENT
        state = await update_state_async(lambda s, b=boredom: s.update({"boredom": b}))
    elif boredom != 0:
        # 離開允許時段（例如半夜）就不累積、也歸零，避免半夜補累積的無聊值在隔天早上一次爆發
        boredom = 0
        state = await update_state_async(lambda s: s.update({"boredom": 0}))

    last_care_time_str = state.get("last_proactive_care_time")
    cooldown_ok = True
    if last_care_time_str:
        try:
            last_care_time = datetime.fromisoformat(last_care_time_str)
            cooldown_ok = (now - last_care_time) >= timedelta(minutes=PROACTIVE_CARE_COOLDOWN_MINUTES)
        except ValueError:
            cooldown_ok = True

    if in_active_window and last_msg_timestamp and boredom >= boredom_target and cooldown_ok:
        target_channel_id = get_safe_target_channel(state)
        if target_channel_id:
            channel = bot.get_channel(int(target_channel_id))
            if channel:
                try:
                    last_msg_time = datetime.fromisoformat(last_msg_timestamp)
                    idle_hours = (now - last_msg_time).total_seconds() / 3600

                    # 記憶提取：撈過去可能未完的話題 + 長效備忘錄
                    topic_context = await query_rag_memory_async(
                        "最近聊過什麼、主人在意或還沒說完的事", target_channel_id, top_k=2
                    )
                    user_profile = load_json(PROFILE_FILE, [])
                    profile_text = "\n".join([f"- {item}" for item in user_profile]) if user_profile else "- 無特定備忘。"

                    persona = load_persona()
                    opener_samples = persona.get("few_shot_examples", [])[:3]
                    opener_examples_text = "\n".join(
                        [f"（語氣參考，不要照抄）：「{ex['diana']}」" for ex in opener_samples]
                    )

                    care_prompt = f"""你現在必須完全沉浸扮演角色「{persona.get('name', '黛安娜')}」，一位{persona.get('age_style', '8~10歲小女孩')}機器人。

【當前情境】
現在時間：{now.strftime('%H:%M')}
主人已經大約 {idle_hours:.1f} 小時沒有說話、頻道很安靜了。

【可能有用的過去回憶／備忘，若與現在無關可以直接忽略，不要硬提】
{topic_context}
{profile_text}

【任務】
主動打破沉默、跟主人說一句話。
- 如果上面的回憶或備忘裡有具體、還沒問完/沒說完的事，優先延續、關心那件事的後續，或藉此延伸一個新話題。
- 如果沒有相關線索，就隨意抱怨肚子餓、無聊、想問主人在忙什麼之類的生活化話題。

【嚴格格式規範】
1. 只能用生活化、口語化的短句開場，例如「欸」「主人」「誒等等」「那個」「唉」；絕對禁止「請問」「有什麼我可以幫忙的嗎」這種助理／客服語氣開場，也不能像在發表演講或做總結。
2. 整段話不超過兩句。
3. 嚴禁堆疊 Emoji（最多 1 個，或完全不用）。
4. 嚴禁文青、抽象修辭。
{opener_examples_text}
"""
                    res_care = await call_llm(MODEL_NAME, [{"role": "user", "content": care_prompt}], 0.8)
                    care_msg = res_care.choices[0].message.content.strip()
                    await channel.send(f"*(小跑步跑過來，拉拉主人的袖子)*\n{care_msg}")

                    def _reset_boredom(s, care_time=now.isoformat()):
                        s["last_proactive_care_time"] = care_time
                        s["boredom"] = 0
                        s["boredom_target"] = random.randint(BOREDOM_TRIGGER_MIN, BOREDOM_TRIGGER_MAX)
                    state = await update_state_async(_reset_boredom)

                    print(f"💖 [無聊值觸發] 無聊值 {boredom} >= 門檻 {boredom_target}，已發送主動關心訊息至頻道 {target_channel_id}")
                except Exception as e:
                    print(f"❌ 主動關心發送失敗: {e}")
        elif state.get("system_notify_channel_id") or state.get("last_active_channel_id"):
            print("🔕 [主動關懷略過] 目標頻道在忽略清單中，本次不推播。")

# ---------------------------------------------------------
# 6. 斜線指令 (Slash Commands) 註冊區
# ---------------------------------------------------------

class UpdatePasswordModal(discord.ui.Modal, title='系統管理員驗證'):
    password_input = discord.ui.TextInput(
        label='請輸入管理員密碼',
        style=discord.TextStyle.short,
        placeholder='輸入密碼以授權更新...',
        required=True
    )

    def __init__(self, update_type: str, file: discord.Attachment):
        super().__init__()
        self.update_type = update_type
        self.file_attachment = file

    async def on_submit(self, interaction: discord.Interaction):
        if self.password_input.value != get_admin_password():
            await interaction.response.send_message("❌ 密碼錯誤，拒絕更新存取！", ephemeral=True)
            return

        old_dir = "./old"
        if not os.path.exists(old_dir):
            os.makedirs(old_dir)

        now = datetime.now()
        yymm = now.strftime("%y%m")

        seq = 1
        prefix = f"{self.update_type}_{yymm}."
        for filename in os.listdir(old_dir):
            if filename.startswith(prefix):
                try:
                    match = re.search(rf'{prefix}(\d+)\.', filename)
                    if match:
                        file_seq = int(match.group(1))
                        if file_seq >= seq:
                            seq = file_seq + 1
                except Exception:
                    pass

        new_version = f"{yymm}.{seq}"
        ext = ".py" if self.update_type == "bot" else ".json"
        archive_filename = f"{self.update_type}_{new_version}{ext}"
        archive_path = os.path.join(old_dir, archive_filename)
        
        active_filename = "bot.py" if self.update_type == "bot" else PERSONA_FILE

        await interaction.response.defer()
        await self.file_attachment.save(archive_path)

        shutil.copy2(archive_path, active_filename)

        vers = load_versions()
        vers[self.update_type] = new_version
        save_versions(vers)

        followup_note = ""
        # 當更新的是 bot.py 本體，代表這次許願池 (upgrade.txt) 裡的需求已經被實作並部署，
        # 將目前使用中的 upgrade.txt 歸檔為 updates/done/done{YYMMDD}.txt，讓 /upgrade 重新從空白開始累積。
        if self.update_type == "bot":
            archived_upgrade_path = archive_done_file(UPGRADE_FILE, UPDATE_DONE_DIR, now)
            if archived_upgrade_path:
                followup_note = f"\n📋 本次已消化的許願清單已歸檔至 `{archived_upgrade_path}`。"

        await interaction.followup.send(
            f"✅ 更新成功！已接收並覆寫新版本 `{new_version}` (備份至 `{archive_path}`)。{followup_note}\n系統將在 3 秒後重新啟動..."
        )

        loop = asyncio.get_running_loop()
        loop.call_later(3, lambda: os.execv(sys.executable, ['python'] + sys.argv))

@bot.tree.command(name="help", description="顯示黛安娜 Bot 的指令清單")
async def help_command(interaction: discord.Interaction):
    help_menu = """🤖 **黛安娜 Bot 指令目錄**
---------------------------------
* `/help`：顯示此指令清單。
* `/growth_log`：查看黛安娜最新的自我成長與演化日誌。
* `/what_to_eat`：不知道吃什麼嗎？讓黛安娜為主人抽籤推薦餐點！
* `/ignore_channel`：設定與管理忽略頻道清單 (需管理員密碼)。
* `/set_notify_channel`：將目前頻道設定為系統通知頻道 (需管理員密碼)。
* `/profile_list`：查看主人的長效備忘錄。
* `/profile_delete <編號> <密碼>`：刪除指定備忘錄條目 (需管理員密碼)。
* `/remind <時間> <內容>`：設定定時提醒。
* `/upgrade <功能描述>`：紀錄你希望 Bot 未來新增的功能或修改建議。
* `/user <密碼>`：設定自己為黛安娜的專屬主人，其餘成員將被視為訪客。
* 聊天時提到「幾點要做什麼」，黛安娜會主動詢問提醒；忙碌未對話時，黛安娜也會主動關心主人喔！
* `/favor`：查看目前好感度數值。
* `/reset <密碼>`：清空短期記憶 (需管理員密碼)。
* `/deepreset <密碼>`：完全重置所有記憶 (需管理員密碼)。
* `/models`：查詢可用模型。
* `/switch <模型名稱>`：切換模型 (僅限主人)。
* `/set_temp <數值>`：調整模型對話溫度 (預設 0.75)。
* `/set_thinking <數量>`：調整模型思考長度上限 (預設 256)。
* `/update <類型> <檔案>`：動態更新系統檔案，將彈出密碼驗證視窗並自動重啟。
* `/restart <密碼>`：手動重新啟動系統 (需管理員密碼)。
* `/版本`：查看當前系統各模組的運行版本。
* `/admin <舊密碼> <新密碼>`：修改管理員密碼。"""
    await interaction.response.send_message(help_menu)

@bot.tree.command(name="ignore_channel", description="管理黛安娜忽略不回應的頻道 (需管理員密碼)")
@app_commands.describe(
    action="選擇操作 (add: 新增當前頻道, remove: 移除當前頻道, list: 查看清單)",
    password="管理員密碼"
)
@app_commands.choices(action=[
    app_commands.Choice(name="➕ 新增當前頻道至忽略清單", value="add"),
    app_commands.Choice(name="➖ 從忽略清單移除當前頻道", value="remove"),
    app_commands.Choice(name="📋 查看目前忽略頻道清單", value="list")
])
async def ignore_channel_command(interaction: discord.Interaction, action: str, password: str):
    if password != get_admin_password():
        await interaction.response.send_message("❌ 密碼錯誤，拒絕存取！", ephemeral=True)
        return
        
    state = load_json(STATE_FILE, {})
    ignored = state.get("ignored_channels", [])
    ch_id = str(interaction.channel_id)

    if action == "add":
        if ch_id not in ignored:
            ignored.append(ch_id)
            state["ignored_channels"] = ignored
            save_json(STATE_FILE, state)
            await interaction.response.send_message(f"✅ 已將頻道 <#{ch_id}> 加入忽略清單，黛安娜將不再回應此頻道的訊息！")
        else:
            await interaction.response.send_message(f"ℹ️ 頻道 <#{ch_id}> 已經在忽略清單中了。")
    elif action == "remove":
        if ch_id in ignored:
            ignored.remove(ch_id)
            state["ignored_channels"] = ignored
            save_json(STATE_FILE, state)
            await interaction.response.send_message(f"✅ 已將頻道 <#{ch_id}> 移出忽略清單！")
        else:
            await interaction.response.send_message(f"ℹ️ 頻道 <#{ch_id}> 不在忽略清單中。")
    elif action == "list":
        if not ignored:
            await interaction.response.send_message("📋 目前沒有設定任何忽略頻道。")
        else:
            ch_list = "\n".join([f"- <#{cid}> (`{cid}`)" for cid in ignored])
            await interaction.response.send_message(f"📋 **目前忽略的頻道清單**：\n{ch_list}")

@bot.tree.command(name="set_notify_channel", description="將目前頻道設定為系統通知頻道 (需管理員密碼)")
@app_commands.describe(password="管理員密碼")
async def set_notify_channel_command(interaction: discord.Interaction, password: str):
    if password != get_admin_password():
        await interaction.response.send_message("❌ 密碼錯誤，拒絕存取！", ephemeral=True)
        return

    ch_id = str(interaction.channel_id)
    state = load_json(STATE_FILE, {})
    state["system_notify_channel_id"] = ch_id
    save_json(STATE_FILE, state)
    await interaction.response.send_message(f"✅ 已將頻道 <#{ch_id}> 設定為黛安娜的系統通知頻道！ (晨間日誌與通知將優先推播至此)")

@bot.tree.command(name="user", description="設定自己為黛安娜的專屬主人 (需管理員密碼)")
@app_commands.describe(password="管理員密碼")
async def user_command(interaction: discord.Interaction, password: str):
    if password != get_admin_password():
        await interaction.response.send_message("❌ 密碼錯誤，拒絕存取！", ephemeral=True)
        return
        
    set_master_id(interaction.user.id)
    await interaction.response.send_message(f"✅ 設定成功！黛安娜以後會專屬認得主人 <@{interaction.user.id}> 囉！其餘的人都會被當作訪客對待。")

@bot.tree.command(name="upgrade", description="新增功能許願池，將想新增的功能記錄下來")
@app_commands.describe(feature="想要新增的功能描述")
async def upgrade_command(interaction: discord.Interaction, feature: str):
    try:
        async with get_file_lock(UPGRADE_FILE):
            os.makedirs(UPDATE_DIR, exist_ok=True)
            with open(UPGRADE_FILE, "a", encoding="utf-8") as f:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{now_str}] {feature}\n")
        await interaction.response.send_message(f"✅ 已將新功能需求紀錄至許願池：\n`{feature}`")
    except Exception as e:
        await interaction.response.send_message(f"❌ 記錄失敗：{e}", ephemeral=True)

@bot.tree.command(name="what_to_eat", description="當主人肚子餓又不知道吃什麼時，讓黛安娜幫忙推薦食物菜單！")
@app_commands.describe(category="你想選擇的類型 (消暑涼爽 / 在家簡單煮 / 外送推薦 / 隨機抽籤)")
@app_commands.choices(category=[
    app_commands.Choice(name="❄️ 消暑涼爽（涼麵、冰品、涼拌）", value="cold"),
    app_commands.Choice(name="🏠 在家簡單煮（水餃、乾拌麵、升級泡麵）", value="easy"),
    app_commands.Choice(name="🛵 外送好選擇（便當、漢堡速食、健康餐）", value="delivery"),
    app_commands.Choice(name="🎲 黛安娜隨機特別推薦", value="random")
])
async def what_to_eat_command(interaction: discord.Interaction, category: str = "random"):
    menu_database = {
        "cold": [
            "日式麻醬涼麵 🍜（天氣太熱吃這個最開胃了！）",
            "鮮蝦雞絲沙拉 🥗（清爽又少負擔喔！）",
            "涼拌豆腐皮蛋配白飯 🍚（簡單又冰涼爽口！）",
            "水果冷麵 🍎（酸酸甜甜很消暑！）"
        ],
        "easy": [
            "豪華特製泡麵 🍜（記得要加一顆蛋和一點青菜喔，這樣比較營養！）",
            "手工水餃 🥟（滾水煮幾分鐘就能吃，不用跑出門吹熱風！）",
            "古早味麻醬乾拌麵 🍝（簡單煮又好吃！）",
            "起司咖哩微波飯 🍛（方便又香氣十足！）"
        ],
        "delivery": [
            "多汁炸雞與漢堡套餐 🍔（太熱不想出門就叫外送吹冷氣吃！）",
            "日式豬排定食便當 🍱（飽足感滿滿！）",
            "健康舒肥雞胸肉餐盒 🥗（好吃又健康！）",
            "夏威夷披薩 🍕（可以跟黛安娜一起分著吃！）"
        ]
    }
    
    if category == "random":
        all_items = [item for sublist in menu_database.values() for item in sublist]
        chosen = random.choice(all_items)
    else:
        chosen = random.choice(menu_database.get(category, menu_database["cold"]))
        
    msg = (
        f"🍽️ **黛安娜的美食推薦** ✨\n\n"
        f"*(歪頭思考了一下，小跑步跑過來拉拉你的袖子)*\n"
        f"主人主人！黛安娜幫你想好今天可以吃什麼了喔：\n"
        f"👉 **【{chosen}】**\n\n"
        f"天氣熱熱的，主人要記得多喝水，不要餓肚子喔！"
    )
    await interaction.response.send_message(msg)

@bot.tree.command(name="growth_log", description="查看黛安娜最新的演化與成長日誌 (log.txt)")
async def growth_log_command(interaction: discord.Interaction):
    if not os.path.exists(LOG_FILE):
        await interaction.response.send_message("🌱 黛安娜目前還沒有新的成長日記喔，等明天早上看看吧！", ephemeral=True)
        return
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            await interaction.response.send_message("🌱 成長日記還是空白的喔！", ephemeral=True)
            return
            
        msg = (
            f"📖 **黛安娜的成長演化筆記**\n"
            f"```text\n"
            f"{content}\n"
            f"```"
        )
        await interaction.response.send_message(msg)
    except Exception as e:
        await interaction.response.send_message(f"❌ 讀取日記失敗: {e}", ephemeral=True)

@bot.tree.command(name="admin", description="修改系統管理員密碼")
@app_commands.describe(old_password="目前的密碼", new_password="欲設定的新密碼")
async def admin_command(interaction: discord.Interaction, old_password: str, new_password: str):
    if old_password != get_admin_password():
        await interaction.response.send_message("❌ 舊密碼錯誤，拒絕修改！", ephemeral=True)
        return
    
    set_admin_password(new_password)
    await interaction.response.send_message("✅ 管理員密碼已成功更新！", ephemeral=True)

@bot.tree.command(name="版本", description="查看當前 Bot 執行檔與角色設定檔的版本")
async def version_command(interaction: discord.Interaction):
    vers = load_versions()
    msg = (f"🏷️ **當前系統版本狀態**\n"
           f"- 🤖 Bot 執行檔 (`bot.py`)：`{vers.get('bot', '未知')}`\n"
           f"- 🎭 角色設定檔 (`character_persona.json`)：`{vers.get('character_persona', '未知')}`")
    await interaction.response.send_message(msg)

@bot.tree.command(name="update", description="上傳新檔案並自動備份、更新與重啟系統 (需管理員權限)")
@app_commands.describe(
    update_type="選擇要更新的類型",
    file="請上傳新的檔案 (.py 或 .json)"
)
@app_commands.choices(update_type=[
    app_commands.Choice(name="🤖 Bot 執行檔 (bot.py)", value="bot"),
    app_commands.Choice(name="🎭 角色設定檔 (character_persona.json)", value="character_persona")
])
async def update_command(interaction: discord.Interaction, update_type: str, file: discord.Attachment):
    await interaction.response.send_modal(UpdatePasswordModal(update_type, file))

@bot.tree.command(name="restart", description="重新啟動系統 (需管理員權限)")
@app_commands.describe(password="管理員密碼")
async def restart_command(interaction: discord.Interaction, password: str):
    if password != get_admin_password():
        await interaction.response.send_message("❌ 密碼錯誤，拒絕存取！", ephemeral=True)
        return
        
    await interaction.response.send_message("🔄 系統正在重新啟動中...")
    loop = asyncio.get_running_loop()
    loop.call_later(2, lambda: os.execv(sys.executable, ['python'] + sys.argv))

@bot.tree.command(name="favor", description="查看黛安娜對主人的好感度")
async def favor_command(interaction: discord.Interaction):
    channel_id = str(interaction.channel_id)
    score = get_favorability(channel_id)
    await interaction.response.send_message(f"💖 黛安娜目前對主人的好感度是：`{score} / 100`！")

async def remind_time_autocomplete(interaction: discord.Interaction, current: str):
    now = datetime.now()
    def _future(dt):
        return dt if dt > now else dt + timedelta(days=1)
    quick_options = [
        ("30 分鐘後", now + timedelta(minutes=30)),
        ("1 小時後", now + timedelta(hours=1)),
        ("今晚 20:00", _future(now.replace(hour=20, minute=0, second=0, microsecond=0))),
        ("明天早上 09:00", (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)),
    ]
    choices = []
    if current.strip():
        choices.append(app_commands.Choice(name=f"直接使用輸入值：{current.strip()}", value=current.strip()))
    for label, dt in quick_options:
        value = dt.strftime("%m-%d %H:%M")
        choices.append(app_commands.Choice(name=f"{label}（{value}）", value=value))
    return choices[:25]

@bot.tree.command(name="remind", description="新增定時提醒")
@app_commands.describe(time="時間格式 (MM-DD HH:MM 或 MM/DD HH:MM)，例如 08/11 09:00", content="提醒內容")
@app_commands.autocomplete(time=remind_time_autocomplete)
async def remind_command(interaction: discord.Interaction, time: str, content: str):
    channel_id = str(interaction.channel_id)
    now = datetime.now()

    try:
        time_normalized = time.strip().replace("/", "-")
        target_time = datetime.strptime(time_normalized, "%m-%d %H:%M").replace(year=now.year)
    except ValueError:
        await interaction.response.send_message(
            "❌ 咦？黛安娜看不懂這個時間格式耶……要用 `MM-DD HH:MM` 或 `MM/DD HH:MM` 喔，像是 `08/11 09:00` 這樣，主人再打一次看看？"
        )
        return

    if target_time <= now:
        target_time = target_time.replace(year=now.year + 1)

    reminders = load_reminders()
    reminders.append({
        "channel_id": channel_id,
        "time": target_time.isoformat(),
        "content": content.strip()
    })
    save_reminders(reminders)
    await interaction.response.send_message(
        f"✅ 黛安娜記下來了！會在 `{target_time.strftime('%Y-%m-%d %H:%M')}` 提醒主人：『{content}』！"
    )

@bot.tree.command(name="profile_list", description="查看主人的長效備忘錄")
async def profile_list_command(interaction: discord.Interaction):
    profile_list = load_json(PROFILE_FILE, [])
    if not profile_list:
        await interaction.response.send_message("📌 **目前的長效備忘錄是空的。**")
        return
    msg = "📌 **主人的靜態備忘錄**：\n" + "\n".join([f"`{i+1}.` {item}" for i, item in enumerate(profile_list)])
    await interaction.response.send_message(msg)

@bot.tree.command(name="profile_delete", description="刪除指定編號的備忘錄條目 (需管理員權限)")
@app_commands.describe(number="要刪除的條目編號 (例如 1)", password="管理員密碼")
async def profile_delete_command(interaction: discord.Interaction, number: int, password: str):
    if password != get_admin_password():
        await interaction.response.send_message("❌ 密碼錯誤，拒絕存取！", ephemeral=True)
        return

    profile_list = load_json(PROFILE_FILE, [])
    if 1 <= number <= len(profile_list):
        removed = profile_list.pop(number - 1)
        save_json(PROFILE_FILE, profile_list)
        await interaction.response.send_message(f"🗑️ 已擦除第 `{number}` 項：『{removed}』")
    else:
        await interaction.response.send_message("❌ 無效的條目編號！")

@bot.tree.command(name="reset", description="清空當前頻道的短期對話記憶 (需管理員權限)")
@app_commands.describe(password="管理員密碼")
async def reset_command(interaction: discord.Interaction, password: str):
    if password != get_admin_password():
        await interaction.response.send_message("❌ 密碼錯誤，拒絕存取！", ephemeral=True)
        return

    channel_id = str(interaction.channel_id)
    short_mem = load_json(MEMORY_FILE, {})
    short_mem[channel_id] = []
    save_json(MEMORY_FILE, short_mem)
    await interaction.response.send_message("欸？黛安娜剛才好像打了個瞌睡……我們剛才在聊甚麼呀，主人？")

@bot.tree.command(name="deepreset", description="徹底重置所有記憶包含短期與長期 RAG (需管理員權限)")
@app_commands.describe(password="管理員密碼")
async def deepreset_command(interaction: discord.Interaction, password: str):
    if password != get_admin_password():
        await interaction.response.send_message("❌ 密碼錯誤，拒絕存取！", ephemeral=True)
        return

    channel_id = str(interaction.channel_id)
    short_mem = load_json(MEMORY_FILE, {})
    short_mem[channel_id] = []
    save_json(MEMORY_FILE, short_mem)
    clear_rag_memory(channel_id)
    await interaction.response.send_message("掃 *(記憶核心重置完成)* 誒……？主人，你是誰呀？")

@bot.tree.command(name="models", description="查詢目前 LM Studio 運行的模型清單")
async def models_command(interaction: discord.Interaction):
    try:
        models_list = await client.models.list()
        available = [m.id for m in models_list.data]
        msg = "⚙️ **LM Studio 可用模型列表**：\n" + "\n".join([f"- `{m}`" for m in available])
        await interaction.response.send_message(msg)
    except Exception as e:
        await interaction.response.send_message(f"❌ 無法讀取模型清單: {e}")

async def model_name_autocomplete(interaction: discord.Interaction, current: str):
    choices = []
    web_api_name = "gemini-3.5-flash-lite"
    if current.lower() in web_api_name.lower() or current.lower() in "web api":
        choices.append(app_commands.Choice(name=f"🌐 Web API ({web_api_name})", value=web_api_name))
        
    try:
        models_list = await client.models.list()
        available = [m.id for m in models_list.data]
    except Exception:
        available = []

    filtered = [name for name in available if current.lower() in name.lower()]
    for name in filtered[:24]:
        choices.append(app_commands.Choice(name=name, value=name))
        
    return choices[:25]

@bot.tree.command(name="switch", description="切換 Bot 呼叫的模型標籤")
@app_commands.describe(model_name="模型名稱（會自動列出本機與 Web API 可用的模型）")
@app_commands.autocomplete(model_name=model_name_autocomplete)
async def switch_command(interaction: discord.Interaction, model_name: str):
    master_id = get_master_id()
    if not master_id or str(interaction.user.id) != master_id:
        await interaction.response.send_message("❌ 只有主人可以幫黛安娜切換模型喔！", ephemeral=True)
        return

    global MODEL_NAME
    MODEL_NAME = model_name.strip()
    await interaction.response.send_message(f"✅ 已切換目標模型為：`{MODEL_NAME}`")

@bot.tree.command(name="set_temp", description="調整模型回覆的溫度 (Temperature)")
@app_commands.describe(temp="請輸入 0.0 到 2.0 之間的數值 (預設 0.75)")
async def set_temp_command(interaction: discord.Interaction, temp: float):
    global CHAT_TEMPERATURE
    if 0.0 <= temp <= 2.0:
        CHAT_TEMPERATURE = temp
        await interaction.response.send_message(f"✅ 已將對話溫度 (Temperature) 調整為：`{CHAT_TEMPERATURE}`")
    else:
        await interaction.response.send_message("❌ 請輸入 0.0 到 2.0 之間的有效數值！", ephemeral=True)

@bot.tree.command(name="set_thinking", description="調整思考長度 (Max Thinking Tokens)")
@app_commands.describe(tokens="輸入新的 token 數量上限 (預設 256)")
async def set_thinking_command(interaction: discord.Interaction, tokens: int):
    global MAX_THINKING_TOKENS
    if tokens > 0:
        MAX_THINKING_TOKENS = tokens
        await interaction.response.send_message(f"✅ 已將思考長度上限調整為：`{MAX_THINKING_TOKENS}` tokens")
    else:
        await interaction.response.send_message("❌ Token 數量必須大於 0！", ephemeral=True)

# ---------------------------------------------------------
# 7. 事件監聽與主對話處理
# ---------------------------------------------------------
@bot.event
async def on_ready():
    print(f'🤖 黛安娜 全功能 Bot 已上線：{bot.user}')
    try:
        synced = await bot.tree.sync()
        print(f"✅ 成功向 Discord 同步 {len(synced)} 個斜線指令！")
    except Exception as e:
        print(f"❌ 指令同步失敗: {e}")

    master_id = get_master_id()
    if master_id:
        try:
            master_user = await bot.fetch_user(int(master_id))
            await master_user.send("☀️ 主人，黛安娜已經成功啟動並連上線囉！")
            print("✅ 已成功向主人發送啟動通知。")
        except Exception as e:
            print(f"❌ 無法向主人發送啟動私訊: {e}")

    if not background_maintenance_task.is_running():
        background_maintenance_task.start()

@bot.event
async def on_message(message):
    global MODEL_NAME, CHAT_TEMPERATURE, MAX_THINKING_TOKENS

    # 過濾掉自己、所有其他 bot / webhook (例如其他機器人的自動推播、系統通知)，以及斜線指令
    if message.author.bot or message.content.startswith("/"):
        return

    channel_id = str(message.channel.id)

    # 需求 1：檢查當前頻道是否在忽略清單中（忽略清單頻道完全不處理、不記錄任何狀態）
    state = await load_json_async(STATE_FILE, {})
    if is_channel_ignored(channel_id, state):
        return

    msg_receive_time = datetime.now()
    clean_prompt = re.sub(r'<@&?!?\d+>', '', message.content).strip()

    def _mark_active(s):
        s["last_message_time"] = datetime.now().isoformat()
        s["last_active_channel_id"] = channel_id
        s["idle_summarized"] = False
        # 只要主人（或訪客）一開口，無聊值就歸零，重新隨機抽一個下次觸發門檻
        s["boredom"] = 0
        s["boredom_target"] = random.randint(BOREDOM_TRIGGER_MIN, BOREDOM_TRIGGER_MAX)
    state = await update_state_async(_mark_active)

    short_mem = await load_json_async(MEMORY_FILE, {})
    if channel_id not in short_mem:
        short_mem[channel_id] = []

    current_fav = get_favorability(channel_id)

    week_days = ["日", "一", "二", "三", "四", "五", "六"]
    now_obj = datetime.now()
    now_str = f"{now_obj.year}年{now_obj.month}月{now_obj.day}日 星期{week_days[now_obj.weekday() if now_obj.weekday() != 6 else 0]} {now_obj.strftime('%H:%M')}"

    user_profile = load_json(PROFILE_FILE, [])
    profile_text = "\n".join([f"- {item}" for item in user_profile]) if user_profile else "- 無特定備忘。"

    memory_keywords = ["記得", "想起", "之前", "上次", "說過", "是誰", "哪裡", "過嗎", "幾號"]
    should_query_rag = any(kw in clean_prompt for kw in memory_keywords)

    if should_query_rag:
        await message.channel.send("*(歪頭想了想)* 誒……讓黛安娜想一下喔，等等告訴主人！")
        retrieved_context = await query_rag_memory_async(clean_prompt, channel_id)
    else:
        retrieved_context = "無（當前對話無需檢索過往回憶）。"

    master_id = get_master_id()
    if master_id:
        is_master = (str(message.author.id) == master_id)
    else:
        is_master = True
        
    user_name = message.author.display_name

    current_system_prompt = build_system_prompt(
        current_time=now_str,
        favor_score=current_fav,
        static_profile=profile_text,
        retrieved_memories=retrieved_context,
        user_name=user_name,       
        is_master=is_master        
    )

    has_image = False
    llm_turn_payload = []
    if message.attachments:
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                has_image = True
                img_bytes = await attachment.read()
                base64_image = base64.b64encode(img_bytes).decode('utf-8')
                llm_turn_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })

    text_value = clean_prompt if clean_prompt else "（傳送了一張圖片）"
    llm_turn_payload.append({"type": "text", "text": text_value})

    raw_history_text = f"（傳送了一張圖片）{text_value}" if has_image else text_value
    if not is_master:
        history_text = f"[{user_name} 說]: {raw_history_text}"
    else:
        history_text = raw_history_text
        
    history_turn_payload = [{"type": "text", "text": history_text}]

    short_mem[channel_id].append({"role": "user", "content": history_turn_payload})
    if len(short_mem[channel_id]) > MAX_SHORT_TERM:
        short_mem[channel_id] = short_mem[channel_id][-MAX_SHORT_TERM:]

    async with get_file_lock(MEMORY_FILE):
        await save_json_async(MEMORY_FILE, short_mem)

    messages_payload = (
        [{"role": "system", "content": current_system_prompt}]
        + short_mem[channel_id][:-1]
        + [{"role": "user", "content": llm_turn_payload}]
    )

    current_target_model = MODEL_NAME if is_master else "google/gemma-4-e2b"

    async with message.channel.typing():
        try:
            response = await call_llm(
                model=current_target_model, 
                messages=messages_payload, 
                temperature=CHAT_TEMPERATURE, 
                max_thinking_tokens=MAX_THINKING_TOKENS
            )
            reply = response.choices[0].message.content.strip()

            await message.channel.send(reply)

            async with get_file_lock(MEMORY_FILE):
                latest_short_mem = await load_json_async(MEMORY_FILE, {})
                if channel_id not in latest_short_mem:
                    latest_short_mem[channel_id] = []
                latest_short_mem[channel_id].append({"role": "assistant", "content": reply})
                if len(latest_short_mem[channel_id]) > MAX_SHORT_TERM:
                    latest_short_mem[channel_id] = latest_short_mem[channel_id][-MAX_SHORT_TERM:]
                await save_json_async(MEMORY_FILE, latest_short_mem)

            print(f"✅ 黛安娜回應: {reply}")

            duration = (datetime.now() - msg_receive_time).total_seconds()
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else "未知"
            print(f"📊 {duration:.2f}秒-{tokens_used}-{current_target_model}")

            remind_times = detect_time_mention(clean_prompt, datetime.now())
            if remind_times:
                view = ReminderConfirmView(
                    channel_id=channel_id,
                    remind_times=remind_times,
                    content=clean_prompt[:80]
                )
                
                if len(remind_times) >= 2:
                    msg_text = (f"對了主人，既然沒有指定具體時間，黛安娜要不要在 "
                                f"`{remind_times[0].strftime('%m-%d %H:%M')}`(前晚睡前) 和 "
                                f"`{remind_times[1].strftime('%m-%d %H:%M')}`(當天早上) 提醒你這件事呀？")
                else:
                    msg_text = (f"對了主人，黛安娜要不要在 `{remind_times[0].strftime('%m-%d %H:%M')}`"
                                f"（提前 30 分鐘）提醒你這件事呀？")
                                
                await message.channel.send(msg_text, view=view)

        except Exception as e:
            print(f"❌ 呼叫失敗: {e}")
            await message.channel.send(f"嗚……黛安娜好像連不上訊號了，是不是哪裡壞掉了呀？(錯誤: {e})")

bot.run(DISCORD_TOKEN)