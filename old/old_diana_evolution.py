"""
diana_evolution.py
==================
黛安娜 (Diana) 自我成長與演化腳本 (手動選擇模型與防頻率限制版)
- 透過 Google Gemini API (支援 API Key 讀取)
- 初次啟動時向 Google 伺服器請求可用模型清單，由使用者手動選擇並記錄在 google API key.txt 第二行
- 讀取當前對話歷史、RAG 長期記憶庫、好感度狀態、舊版人設與程式碼
- 每天定時排程 (或獨立執行一次) 進行兩大演化模組：
  1. Persona Evolution (角色心智與人設成長微調 -> character_persona.json)
  2. Feature Evolution (功能導向新增/微調，無需求保持現狀 -> bot.py)
- 自動備份至 ./old 資料夾並依照 YYMM.seq 版本號歸檔更新 version.json
- 自動生成「成長日誌 (log.txt)」，供 Bot 在隔天早上讀取並主動向主人匯報自己的成長
"""

import os
import sys
import json
import re
import time
import shutil
import schedule
from datetime import datetime
import chromadb
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 1. 檔案路徑與基礎常數配置
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GOOGLE_API_KEY_FILE = os.path.join(BASE_DIR, "google API key.txt")
PERSONA_FILE = os.path.join(BASE_DIR, "character_persona.json")
BOT_FILE = os.path.join(BASE_DIR, "bot.py")
VERSION_FILE = os.path.join(BASE_DIR, "version.json")
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")
PROFILE_FILE = os.path.join(BASE_DIR, "user_profile.json")
FAVOR_FILE = os.path.join(BASE_DIR, "favorability.json")
LOG_FILE = os.path.join(BASE_DIR, "log.txt")
OLD_DIR = os.path.join(BASE_DIR, "old")
CHROMA_DIR = os.path.join(BASE_DIR, "diana_memory_db")

# ---------------------------------------------------------
# 2. 工具函式：金鑰與模型設定、檔案讀寫、版本控制、日誌管理
# ---------------------------------------------------------
def get_api_config() -> tuple[str, str]:
    """讀取 google API key 與模型設定，若無模型則啟動互動選擇程序"""
    # 支援使用環境變數
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model_name = os.environ.get("GEMINI_MODEL_NAME", "").strip()

    key_path = GOOGLE_API_KEY_FILE
    if not os.path.exists(key_path):
        # 兼容無副檔名的情況
        alt_path = os.path.join(BASE_DIR, "google API key")
        if os.path.exists(alt_path):
            key_path = alt_path

    if os.path.exists(key_path):
        with open(key_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.read().splitlines() if line.strip()]
        
        if not api_key and lines:
            api_key = lines[0]
        if not model_name and len(lines) > 1:
            model_name = lines[1]

    if not api_key:
        print(f"❌ 找不到有效的 Google API Key！請建立 '{GOOGLE_API_KEY_FILE}' 並在第一行貼入金鑰。")
        sys.exit(1)

    # 如果檔案中沒有記錄模型名稱，則向伺服器查詢並讓使用者選擇
    if not model_name:
        print("\n🔍 初次啟動或未設定模型，正在向 Google 伺服器查詢可用模型...")
        try:
            client = genai.Client(api_key=api_key)
            models = list(client.models.list())
            available_models = [m.name.replace("models/", "") for m in models if "gemini" in m.name]
            
            print("\n【請選擇您要用於演化的模型】")
            print("💡 建議選擇有免費額度的穩定模型，例如 gemini-1.5-flash 或 gemini-1.5-pro")
            for i, m_name in enumerate(available_models):
                print(f"{i+1}. {m_name}")
            
            while True:
                try:
                    choice = int(input("\n👉 請輸入模型編號 (數字): "))
                    if 1 <= choice <= len(available_models):
                        model_name = available_models[choice-1]
                        break
                    else:
                        print("❌ 輸入超出範圍，請重試。")
                except ValueError:
                    print("❌ 請輸入有效的數字編號。")
            
            # 將 API Key 和選擇的模型寫回檔案
            with open(key_path, "w", encoding="utf-8") as f:
                f.write(f"{api_key}\n{model_name}\n")
            print(f"✅ 設定完成！模型 [{model_name}] 已成功儲存至 {key_path} 的第二行。\n")
            
        except Exception as e:
            print(f"❌ 查詢模型清單失敗: {e}")
            print("⚠️ 將自動使用預設模型 'gemini-1.5-flash'")
            model_name = "gemini-1.5-flash"
            with open(key_path, "w", encoding="utf-8") as f:
                f.write(f"{api_key}\n{model_name}\n")

    return api_key, model_name

def load_json(filepath, default_val):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 讀取 {filepath} 失敗: {e}，返回預設值")
            return default_val
    return default_val

def save_json(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_next_version(update_type: str) -> str:
    """計算符合 YYMM.seq 的下一版本號"""
    os.makedirs(OLD_DIR, exist_ok=True)
    now = datetime.now()
    yymm = now.strftime("%y%m")
    seq = 1
    prefix = f"{update_type}_{yymm}."
    
    for filename in os.listdir(OLD_DIR):
        if filename.startswith(prefix):
            try:
                match = re.search(rf'{prefix}(\d+)\.', filename)
                if match:
                    file_seq = int(match.group(1))
                    if file_seq >= seq:
                        seq = file_seq + 1
            except Exception:
                pass
    return f"{yymm}.{seq}"

def archive_and_update(update_type: str, new_content: str) -> str:
    """將舊版封存到 old/ 並更新當前運行檔與 version.json，回傳新版本號"""
    new_version = generate_next_version(update_type)
    ext = ".py" if update_type == "bot" else ".json"
    archive_filename = f"{update_type}_{new_version}{ext}"
    archive_path = os.path.join(OLD_DIR, archive_filename)
    
    active_filename = BOT_FILE if update_type == "bot" else PERSONA_FILE
    
    # 備份舊檔案
    if os.path.exists(active_filename):
        shutil.copy2(active_filename, archive_path)
        print(f"📦 已將上一版檔案備份至: {archive_path}")
    
    # 寫入新檔案
    with open(active_filename, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    # 更新 version.json
    vers = load_json(VERSION_FILE, {"bot": "1.0", "character_persona": "1.0"})
    vers[update_type] = new_version
    save_json(VERSION_FILE, vers)
    print(f"🎉 成功更新 [{update_type}] 至版本 {new_version}！")
    return new_version

def write_growth_log(persona_summary: str, code_summary: str, persona_ver: str, code_ver: str):
    """
    寫入成長日誌到 log.txt
    供 bot.py 於隔日早晨主動讀取並以黛安娜的語氣發送給主人
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    log_content = (
        f"【黛安娜成長演化日誌】\n"
        f"演化時間：{now_str}\n"
        f"角色設定版本：{persona_ver or '維持原樣'}\n"
        f"程式功能版本：{code_ver or '維持原樣'}\n"
        f"----------------------------------------\n"
        f"🌱 心智與回憶學習：\n{persona_summary}\n\n"
        f"🛠️ 技能與功能學習：\n{code_summary}\n"
    )
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"📝 成長日誌已寫入: {LOG_FILE}")

# ---------------------------------------------------------
# 3. 資料收集模組 (聚集記憶, 好感度, 近期對話)
# ---------------------------------------------------------
def collect_growth_context() -> dict:
    """彙整所有近期互動與成長記憶資料"""
    context = {}
    
    # 1. 好感度
    fav_data = load_json(FAVOR_FILE, {})
    avg_fav = sum(fav_data.values()) / max(len(fav_data), 1) if fav_data else 50
    context["favorability"] = fav_data
    context["avg_favorability"] = avg_fav

    # 2. 近期短期對話
    memory_data = load_json(MEMORY_FILE, {})
    all_turns = []
    for ch_id, history in memory_data.items():
        for turn in history:
            role = "主人" if turn.get("role") == "user" else "黛安娜"
            content = turn.get("content")
            if isinstance(content, list):
                text_part = next((item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"), "")
            else:
                text_part = str(content)
            all_turns.append(f"{role}: {text_part}")
    context["recent_dialogues"] = "\n".join(all_turns[-30:]) if all_turns else "近期無直接對話記錄。"

    # 3. 長期靜態備忘
    profiles = load_json(PROFILE_FILE, [])
    context["user_profiles"] = "\n".join([f"- {p}" for p in profiles]) if profiles else "無長效記憶標籤。"

    # 4. ChromaDB 向量回憶事件
    rag_docs = []
    if os.path.exists(CHROMA_DIR):
        try:
            chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
            try:
                collection = chroma_client.get_collection(name="diana_memories")
                data = collection.get()
                rag_docs = data.get("documents", [])
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ 讀取 ChromaDB 略過: {e}")
    context["rag_memories"] = "\n".join(rag_docs[-20:]) if rag_docs else "尚無特定向量回憶總結。"

    return context

# ---------------------------------------------------------
# 4. 演化核心 1：角色設定自我微調 (Persona Evolution)
# ---------------------------------------------------------
def evolve_persona(client: genai.Client, context: dict, model_name: str) -> tuple[str, str]:
    """
    進行角色心智微調
    回傳: (更新摘要, 新版本號)
    """
    print(f"\n🌸 [1/2] 正在使用 {model_name} 進行角色人設演化...")
    current_persona = load_json(PERSONA_FILE, {})
    current_persona_str = json.dumps(current_persona, ensure_ascii=False, indent=2)

    prompt = f"""你現在是「黛安娜」的心智成長塑造系統。
黛安娜是一個 8~10 歲實械體小女孩機器人，會隨著跟「主人」的日常相處、互動回憶與好感度，產生真實的心智與性格成長。

【當前角色設定 (character_persona.json)】
```json
{current_persona_str}
```

【近期與主人的互動與記憶數據】
- 當前平均好感度：{context['avg_favorability']} / 100
- 主人長期畫像與備忘：
{context['user_profiles']}
- 近期發生的回憶事件總結 (RAG)：
{context['rag_memories']}
- 近期對話片段：
{context['recent_dialogues']}

【成長微調指引】
1. 黛安娜正在「長大」與「學習」：
   - 根據主人最近分享的事物或日常習慣，微調或擴充 `few_shot_examples`（保留精華，汰換或增加 1~2 筆最能展現近期相處默契的小孩視角真實範例）。
   - 在 `personality_traits` 中增添/微調 1 條展現「她記住了主人教過的事情」的心智細節（但核心小女孩童真、好奇心不變）。
   - 根據好感度階段，微調 `favorability_reactions` 語氣。
2. 保持小女孩的自然與童真，嚴禁將她改造成客套、文青或戀愛語氣。
3. 請回傳 JSON 格式，必須包含以下兩個欄位：
   - "growth_summary": 用 1~2 句繁體中文簡要說明這次黛安娜心智學會了什麼（例如：記住了主人喜歡喝黑咖啡、學會了更體貼主人下班累累的時候）。
   - "updated_persona": 完整且更新後的 character_persona JSON 物件（保留原有鍵名結構）。

請直接輸出合法 JSON，不要包含任何 markdown 說明標籤："""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                response_mime_type="application/json"
            )
        )
        res_json = json.loads(response.text.strip())
        growth_summary = res_json.get("growth_summary", "今天黛安娜更了解主人的一點點習慣了！")
        new_persona_data = res_json.get("updated_persona", res_json)
        
        # 驗證必要欄位
        if "personality_traits" not in new_persona_data or "tone_rules" not in new_persona_data:
            raise ValueError("回傳的 persona 格式缺少關鍵欄位")

        formatted_json = json.dumps(new_persona_data, ensure_ascii=False, indent=2)
        new_ver = archive_and_update("character_persona", formatted_json)
        print("✅ 角色人設演化微調完成！")
        return growth_summary, new_ver
    except Exception as e:
        print(f"❌ 角色人設演化失敗: {e}")
        return f"角色心智演化遇到狀況，保持昨日狀態。(原因: {e})", ""

# ---------------------------------------------------------
# 5. 演化核心 2：功能導向擴充評估 (Feature-Driven Evolution)
# ---------------------------------------------------------
def evolve_code(client: genai.Client, context: dict, model_name: str) -> tuple[str, str]:
    """
    評估並新增功能（保持最小侵入性，無需求時回傳 NO_CHANGE）
    回傳: (更新摘要, 新版本號)
    """
    print(f"\n⚙️ [2/2] 正在使用 {model_name} 評估 Bot 潛在新功能需求與程式更新...")
    if not os.path.exists(BOT_FILE):
        print(f"❌ 找不到 {BOT_FILE}，略過程式碼評估。")
        return "找不到原始程式碼，略過更新。", ""

    with open(BOT_FILE, "r", encoding="utf-8") as f:
        current_code = f.read()

    prompt = f"""你是一位資深的 Python 與 Discord.py 軟體架構師。
請根據黛安娜（Diana）近期與主人的互動情境，評估是否需要為 Bot「新增實用功能」或「微調擴充現有指令」。

【近期主人與黛安娜的對話與記憶概況】
- 長期備忘標籤：
{context.get('user_profiles', '無')}
- 近期對話片段：
{context.get('recent_dialogues', '無')}
- 當前好感度：{context.get('avg_favorability', 50)} / 100

【現有 bot.py 程式碼】
```python
{current_code}
```

【演化核心原則（嚴格遵守）】
1. **主要任務（功能性擴充）**：
   - 觀察主人是否有反覆提及的需求或特定生活情境（例如：想要特定小遊戲、專屬計數器、趣味互動指令、情緒安撫小功能等）。
   - 若有明確價值，可新增對應的 Discord Slash Command (`@bot.tree.command`) 或在訊息監聽中增加特定彩蛋反應。
2. **次要任務（最小侵入性修改，嚴禁過度重構）**：
   - 保持現有的整體架構、全域變數與流程不變。
   - 嚴禁大幅改寫既有的 RAG (ChromaDB)、檔案管理與密碼重啟系統。
   - 僅在新增功能時補足必要的 try-except 防呆，避免因過度優化破壞系統穩定性。
3. **無需求時保持現狀**：
   - 如果評估現有功能已完全滿足需求且近期無新增功能的急迫性，請回傳 NO_CHANGE。

【輸出格式要求】
請回傳 JSON 格式：
- 若無需更新：
  {{"status": "NO_CHANGE", "summary": "目前功能運行穩定，無新增功能需求。"}}
- 若決定新增功能：
  {{"status": "UPDATED", "summary": "簡述本次為黛安娜新增了什麼技能或指令", "code": "完整且可執行的完整 bot.py Python 程式碼"}}"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                response_mime_type="application/json"
            )
        )
        res_json = json.loads(response.text.strip())
        status = res_json.get("status", "NO_CHANGE")
        summary = res_json.get("summary", "今日程式保持原有穩定狀態。")

        if status == "NO_CHANGE":
            print("🛡️ [程式評估完成] 目前功能運行穩定，今日無新增功能需求，保持現有程式碼。")
            return summary, ""

        new_code = res_json.get("code", "").strip()
        if not new_code:
            print("⚠️ 未取得完整程式碼，保持原樣。")
            return "程式碼生成不完整，維持原狀。", ""

        # 清理可能夾帶的 markdown
        if new_code.startswith("```python"):
            new_code = new_code[9:]
        elif new_code.startswith("```"):
            new_code = new_code[3:]
        if new_code.endswith("```"):
            new_code = new_code[:-3]
        new_code = new_code.strip()

        # 語法與編譯安全性檢驗 (Syntax Check)
        try:
            compile(new_code, "<string>", "exec")
        except SyntaxError as se:
            print(f"❌ 新增功能後的程式碼未通過語法檢查 (SyntaxError): {se}，放棄本次更新以確保安全。")
            return f"新功能語法檢查未通過 ({se})，已安全取消更新。", ""

        new_ver = archive_and_update("bot", new_code)
        print("🎉 黛安娜成功習得了新程式功能並完成安全更新！")
        return summary, new_ver
    except Exception as e:
        print(f"❌ 程式自主演化評估失敗: {e}")
        return f"程式演化評估過程異常: {e}，維持原狀。", ""

# ---------------------------------------------------------
# 6. 主要排程與執行入口
# ---------------------------------------------------------
def run_evolution_cycle():
    print(f"\n=======================================================")
    print(f"🌟 [演化週期啟動] 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=======================================================")
    
    try:
        # 讀取金鑰與模型（初次執行會引導選擇）
        api_key, model_name = get_api_config()
        client = genai.Client(api_key=api_key)
        
        # 1. 搜集記憶與對話
        context = collect_growth_context()
        
        # 2. 演化角色人設 (主要心智成長)
        persona_summary, persona_ver = evolve_persona(client, context, model_name)
        
        # 加入緩衝時間，避免 Free Tier 的 TPM (每分鐘 Token 數) 被瞬間塞爆
        print("\n⏳ 為了避免觸發 Google API 頻率限制 (Rate Limit)，暫停 30 秒...")
        time.sleep(30)
        
        # 3. 演化與新增功能 (功能導向，無需求時保持現狀)
        code_summary, code_ver = evolve_code(client, context, model_name)
        
        # 4. 寫入成長日誌到 log.txt 供 Bot 隔天早晨讀取
        write_growth_log(persona_summary, code_summary, persona_ver, code_ver)
        
        # 驗證是否真的成功
        if "異常" in code_summary or "狀況" in persona_summary:
            print(f"\n⚠️ 今日演化循環已結束，但過程中發生了 API 錯誤 (詳見 log.txt)，請檢查額度或網路狀態。\n")
        else:
            print(f"\n✨ 今日演化循環圓滿完成！日誌已儲存至 log.txt，黛安娜又成長了一點點 🌱\n")
            
    except Exception as e:
        print(f"❌ 演化流程發生異常: {e}")

def main():
    print("🚀 黛安娜自我成長與演化引擎 (Diana Evolution Engine) 已啟動")
    print("📌 預設每日凌晨 04:00 自動執行一次演化微調與日誌產出。")
    
    # 確保啟動時就驗證金鑰與模型設定
    get_api_config()
    
    # 支援手動觸發參數
    if len(sys.argv) > 1 and sys.argv[1] in ["--run-now", "-now", "run"]:
        run_evolution_cycle()
        return

    # 設定每日定時排程 (每天凌晨 04:00 執行)
    schedule.every().day.at("04:00").do(run_evolution_cycle)
    
    print("\n⏳ 排程監聽中... (可按 Ctrl + C 結束)")
    while True:
        schedule.run_pending()
        time.sleep(30)

if __name__ == "__main__":
    main()
