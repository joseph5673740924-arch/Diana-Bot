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
from datetime import datetime, timedelta
from chromadb.utils import embedding_functions
from openai import AsyncOpenAI

# ---------------------------------------------------------
# 0. 快速配置區
# ---------------------------------------------------------
TOKEN_FILE = "discord_token.txt"
MODEL_NAME = "local-model"  # 預設搭配 Gemma 4

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
PROFILE_FILE = "user_profile.json"
FAVOR_FILE = "favorability.json"
REMINDER_FILE = "reminders.json"
MEMORY_FILE = "memory.json"
STATE_FILE = "bot_state.json"
PERSONA_FILE = "character_persona.json"
MAX_SHORT_TERM = 8

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

# -----------------------------------------------------------
# 檔案鎖：避免多頻道同時「讀取→修改→寫回」同一個 JSON 檔案時
# 互相覆蓋彼此的變更
# -----------------------------------------------------------
_file_locks = {}

def get_file_lock(filepath):
    if filepath not in _file_locks:
        _file_locks[filepath] = asyncio.Lock()
    return _file_locks[filepath]

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

# -----------------------------------------------------------
# 對話中主動偵測「時間 + 安排」的句子，例如「我晚上8點要開會」
# -----------------------------------------------------------
TIME_MENTION_PATTERN = re.compile(
    r'(?P<day>今天|明天|後天|大後天|下週[一二三四五六日天]|下禮拜[一二三四五六日天])?\s*'
    r'(?P<period>清晨|凌晨|早上|上午|中午|下午|晚上|傍晚)?\s*'
    r'(?:(?P<hour>\d{1,2})\s*(?:[:：]\s*(?P<minute>\d{2})|點\s*(?P<half>半)?))?'
)
PLAN_KEYWORDS = ['要', '得', '需要', '會去', '去', '有空', '有事', '約']

def detect_time_mention(text, now):
    """從使用者訊息裡抓「幾點要做什麼」，回傳需要設定的 datetime 列表。"""
    if not text or not any(kw in text for kw in PLAN_KEYWORDS):
        return None

    for match in TIME_MENTION_PATTERN.finditer(text):
        day_str = match.group('day')
        period_str = match.group('period')
        hour_str = match.group('hour')
        
        if not day_str and not hour_str and not period_str:
            continue
            
        # 基準日：把時間歸零
        target_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # 1. 決定目標日期
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

        # 2. 判斷是否有具體時間或時段
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
            # 有講時段，給予預設時間
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
            # 只有日期，完全沒講時段和時間 -> 預設前晚 22:00 與當天 07:30 提醒
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
    """黛安娜主動詢問要不要設提醒時附上的確認按鈕，120 秒沒人按就自動失效。"""
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
        "relationship": "主人是你唯一的依靠與保護者，關係像女兒與家長一樣，禁止曖昧或戀愛語氣。",
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
# 3. 初始化 Bot (改用 commands.Bot 支援 Slash Commands)
# ---------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

def build_system_prompt(current_time, favor_score, static_profile, retrieved_memories):
    persona = load_persona()
    rules_text = "\n".join([f"- {rule}" for rule in persona.get("tone_rules", [])])

    traits_text = "\n".join([f"- {t}" for t in persona.get("personality_traits", [])])
    speech_text = "\n".join([f"- {s}" for s in persona.get("speech_patterns", [])])

    examples = persona.get("few_shot_examples", [])
    examples_text = ""
    for ex in examples:
        examples_text += f"主人：「{ex['user']}」\n黛安娜：「{ex['diana']}」\n\n"

    reactions = persona.get("favorability_reactions", {})
    if favor_score <= 35:
        current_reaction = reactions.get("low", "")
    elif favor_score <= 70:
        current_reaction = reactions.get("medium", "")
    else:
        current_reaction = reactions.get("high", "")

    return f"""你現在必須完全沉浸並扮演角色「{persona.get('name', '黛安娜')}」。

【角色說話風格與年齡設定】
1. 年齡感：8~10歲的小女孩。請用國小低年級小朋友最自然、簡單直白的口語回答。
2. 背景與關係：{persona.get('background', '')} {persona.get('relationship', '')}
3. 當前時間：{current_time}
4. 當前好感度：{favor_score} / 100（語氣指示：{current_reaction}）

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
        res_fav = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1
        )
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
        res_sum = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.3
        )
        summary = res_sum.choices[0].message.content.strip()
        save_summary_to_rag(summary, channel_id)
        print(f"✅ [2小時閒置記憶歸檔完成]: {summary}")
    except Exception as e:
        print(f"❌ 閒置記憶摘要失敗: {e}")

# ---------------------------------------------------------
# 5. 定時維護任務
# ---------------------------------------------------------
def parse_reminder_time(raw, now):
    """解析提醒時間，支援 ISO 格式與 MM-DD/MM/DD HH:MM 格式。"""
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

    reminders = load_reminders()
    remaining = []
    for item in reminders:
        target_time = parse_reminder_time(item.get("time", ""), now)
        if target_time is None:
            remaining.append(item)
            continue
        if now >= target_time:
            channel = bot.get_channel(int(item["channel_id"]))
            if channel:
                await channel.send(f"⏰ *(拉拉主人的袖子)* 主人！黛安娜提醒你：『{item['content']}』的時間到了喔！")
        else:
            remaining.append(item)
    if len(reminders) != len(remaining):
        save_reminders(remaining)

    state = load_json(STATE_FILE, {})
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
            state["idle_summarized"] = True
            save_json(STATE_FILE, state)

    last_weekly = state.get("last_weekly_clean")
    if now.weekday() == 6 and now.hour == 3 and last_weekly != now.strftime("%Y-%W"):
        try:
            all_rag = collection.get()
            docs = all_rag.get("documents", [])
            if docs:
                all_docs_text = "\n".join(docs)
                prompt = f"請提取出關於主人的『長期習慣、重大喜好、長期設定』，整理成條列式備忘（最多 5 條）：\n{all_docs_text}"
                res = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                weekly_summary = res.choices[0].message.content.strip()
                profiles = load_json(PROFILE_FILE, [])
                profiles.append(f"【每週記憶提煉 ({now.strftime('%m/%d')})】\n{weekly_summary}")
                save_json(PROFILE_FILE, profiles)
                for id_item in all_rag.get("ids", []):
                    collection.delete(ids=[id_item])
            state["last_weekly_clean"] = now.strftime("%Y-%W")
            save_json(STATE_FILE, state)
        except Exception as e:
            print(f"❌ 每週整理失敗: {e}")

# ---------------------------------------------------------
# 6. 斜線指令 (Slash Commands) 註冊區
# ---------------------------------------------------------

@bot.tree.command(name="help", description="顯示黛安娜 Bot 的指令清單")
async def help_command(interaction: discord.Interaction):
    help_menu = """🤖 **黛安娜 Bot 指令目錄**
---------------------------------
* `/help`：顯示此指令清單。
* `/profile_list`：查看主人的長效備忘錄。
* `/profile_delete <編號>`：刪除指定備忘錄條目。
* `/remind <時間> <內容>`：設定定時提醒（支援 MM-DD 或 MM/DD，也可用快選選單）。
* 聊天時提到「幾點要做什麼」或「明天/下週幾要做什麼」，黛安娜也會主動詢問提醒。
* `/favor`：查看目前好感度數值。
* `/reset`：清空短期記憶。
* `/deepreset`：完全重置所有記憶。
* `/models`：查詢可用模型。
* `/switch <模型名稱>`：切換模型。"""
    await interaction.response.send_message(help_menu)

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

@bot.tree.command(name="profile_delete", description="刪除指定編號的備忘錄條目")
@app_commands.describe(number="要刪除的條目編號 (例如 1)")
async def profile_delete_command(interaction: discord.Interaction, number: int):
    profile_list = load_json(PROFILE_FILE, [])
    if 1 <= number <= len(profile_list):
        removed = profile_list.pop(number - 1)
        save_json(PROFILE_FILE, profile_list)
        await interaction.response.send_message(f"🗑️ 已擦除第 `{number}` 項：『{removed}』")
    else:
        await interaction.response.send_message("❌ 無效的條目編號！")

@bot.tree.command(name="reset", description="清空當前頻道的短期對話記憶")
async def reset_command(interaction: discord.Interaction):
    channel_id = str(interaction.channel_id)
    short_mem = load_json(MEMORY_FILE, {})
    short_mem[channel_id] = []
    save_json(MEMORY_FILE, short_mem)
    await interaction.response.send_message("欸？黛安娜剛才好像打了個瞌睡……我們剛才在聊甚麼呀，主人？")

@bot.tree.command(name="deepreset", description="徹底重置所有記憶（短期 + 長期 RAG）")
async def deepreset_command(interaction: discord.Interaction):
    channel_id = str(interaction.channel_id)
    short_mem = load_json(MEMORY_FILE, {})
    short_mem[channel_id] = []
    save_json(MEMORY_FILE, short_mem)
    clear_rag_memory(channel_id)
    await interaction.response.send_message("🧹 *(記憶核心重置完成)* 誒……？主人，你是誰呀？")

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
    try:
        models_list = await client.models.list()
        available = [m.id for m in models_list.data]
    except Exception:
        return []

    filtered = [name for name in available if current.lower() in name.lower()]
    return [app_commands.Choice(name=name, value=name) for name in filtered[:25]]

@bot.tree.command(name="switch", description="切換 Bot 呼叫的模型標籤")
@app_commands.describe(model_name="模型名稱（會自動列出 LM Studio 目前可用的模型）")
@app_commands.autocomplete(model_name=model_name_autocomplete)
async def switch_command(interaction: discord.Interaction, model_name: str):
    global MODEL_NAME
    MODEL_NAME = model_name.strip()
    await interaction.response.send_message(f"✅ 已切換目標模型為：`{MODEL_NAME}`")

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

    if not background_maintenance_task.is_running():
        background_maintenance_task.start()

@bot.event
async def on_message(message):
    global MODEL_NAME

    if message.author == bot.user or message.content.startswith("/"):
        return

    clean_prompt = re.sub(r'<@&?!?\d+>', '', message.content).strip()
    channel_id = str(message.channel.id)

    state = load_json(STATE_FILE, {})
    state["last_message_time"] = datetime.now().isoformat()
    state["idle_summarized"] = False
    save_json(STATE_FILE, state)

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

    current_system_prompt = build_system_prompt(
        current_time=now_str,
        favor_score=current_fav,
        static_profile=profile_text,
        retrieved_memories=retrieved_context
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

    history_text = f"（傳送了一張圖片）{text_value}" if has_image else text_value
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

    async with message.channel.typing():
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages_payload,
                temperature=0.75,
                extra_body={
                    "max_thinking_tokens": 256
                }
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

            # 偵測對話中是否有安排
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
            await message.channel.send(f"主人……黛安娜好像連不上訊號了，是不是哪裡壞掉了呀？(錯誤: {e})")

bot.run(DISCORD_TOKEN)