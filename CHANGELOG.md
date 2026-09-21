# 黛安娜 Bot — 修復與架構調整紀錄 (2026-08-17)

## 🐛 核心 Bug：忽略頻道設定沒有真正生效

**根因**：`ignored_channels` 的檢查只寫在 `on_message`（被動回覆使用者訊息）裡，
但背景任務 `background_maintenance_task` 裡有 3 個「主動推播」的地方完全沒檢查忽略清單：

1. **定時提醒 (區塊 A)**：直接對 `item["channel_id"]` 發送，不管是否被忽略。
2. **晨間成長日誌推播 (區塊 D)**：目標頻道 = `system_notify_channel_id` 或
   `last_active_channel_id`，沒檢查忽略清單。
3. **主動關懷訊息 (區塊 E)**：邏輯同上。

而 `last_active_channel_id` 只要「該頻道曾經不是忽略頻道、且有人講過話」就會被記住，
之後就算你把該頻道加進忽略清單，`last_active_channel_id` 也不會自動清掉
（實測你上傳的 `bot_state.json` 裡，`last_active_channel_id` 正好就是
`ignored_channels` 清單裡的其中一個頻道）。於是只要 `system_notify_channel_id`
沒設或抓不到，區塊 D / E 就會 fallback 回這個「已被你忽略」的頻道，黛安娜就自己冒出來講話。

**修法**：
- 新增 `is_channel_ignored()` 輔助函式，統一判斷邏輯。
- 定時提醒：目標頻道在忽略清單 → 直接捨棄該筆提醒，不發送（並印出 log）。
- 晨間日誌 / 主動關懷：新增 `get_safe_target_channel()`，只要候選頻道被忽略，
  直接視為「沒有可用頻道」，**不會再往下 fallback**，本次跳過推播。

## 🐛 額外發現：沒有過濾其他 Bot / Webhook 訊息

`on_message` 原本只排除 `message.author == bot.user`（自己），**沒有排除其他機器人**。
如果同一個頻道有別的 bot（例如自動推播系統、股票機器人、CI 通知）發訊息，
黛安娜會把它當成一般使用者訊息去回應——這正好符合你說的「自動推播的其他 bot 或系統訊息」。

**修法**：`on_message` 開頭改成 `if message.author.bot or message.content.startswith("/"): return`，
一次擋掉所有機器人／webhook 訊息（不只是自己）。忽略頻道清單仍保留，可以用來擋「連讀都不用讀」的
特定頻道（例如你完全不想讓黛安娜載入該頻道任何狀態的場合）。

## ⚙️ 架構優化

### 1. Race condition：`bot_state.json` 讀寫沒有鎖
`on_message`（幾乎每則訊息都觸發）和 `background_maintenance_task`（每 60 秒一次）
都會對同一個 `bot_state.json` 做「讀取 → 修改 → 整檔寫回」，兩者同時執行時後寫的會覆蓋先寫的。
新增 `update_state_async(mutate_fn)`：用 `asyncio.Lock` 包住整個讀改寫過程，
所有原本 `state[...] = ...; save_json(STATE_FILE, state)` 的地方都改用這個函式。

### 2. 資料夾結構重整
原本除了 `diana_memory_db/`（向量庫）和 `old/`（版本備份）之外，其餘檔案全部散落在根目錄。
調整後：

```
bot.py
config/
  character_persona.json      # 人設檔（更新走 /update persona，一樣會備份進 old/）
data/
  base.json                   # 管理密碼 / 主人 ID
  bot_state.json               # 執行期狀態
  favorability.json
  memory.json                 # 短期對話記憶
  reminders.json
  user_profile.json
  version.json
logs/
  log.txt                      # 使用中的成長日誌（外部流程持續寫入）
  done/
    done260817.txt             # 已推播過的日誌，格式：done{YYMMDD}.txt
updates/
  upgrade.txt                  # 使用中的許願池（/upgrade 指令寫入）
  done/
    done260814.txt             # 已被消化的許願清單
    done260816.txt
diana_memory_db/                # 向量記憶庫（不動）
old/                            # 既有版本備份機制（不動，維持你原本的腳本邏輯）
```

`config/discord_token.txt`、`config/google API key -2.txt` 是新的密鑰存放位置
（因為你上傳時本來就排除了，這兩個檔案在這次交付裡不存在，請自行放進 `config/` 資料夾）。

**向下相容**：`bot.py` 啟動時會呼叫 `migrate_legacy_layout()`，
如果偵測到根目錄還有舊版散落檔案（例如你直接拿舊資料夾覆蓋部署），會自動搬進對應新資料夾，
不需要手動搬檔案。

### 3. 自動更新（upgrade.txt / log.txt）歸檔格式統一
原本兩種檔案的「已使用」歸檔格式不一致：
- `upgrade.txt.done_20260816_1329`（帶完整時間戳、前綴是原檔名）
- `log_20260816_071500.txt`（另一套命名）

新增共用函式 `archive_done_file(source_path, done_dir)`：統一輸出成 `done{YYMMDD}.txt`
(同一天重複歸檔會自動加 `_2` / `_3` 尾碼避免覆蓋)，套用在：

- **log.txt**：晨間日誌推播成功後，從 `logs/log.txt` 歸檔到 `logs/done/done{YYMMDD}.txt`
  （行為不變，只是命名規則統一、且搬到 `logs/done/` 而不是塞進 `old/`）。
- **upgrade.txt**：`/update` 指令上傳新版 `bot.py` 並成功套用重啟時（代表這批許願清單已經
  被實作進去了），自動把 `updates/upgrade.txt` 歸檔到 `updates/done/done{YYMMDD}.txt`，
  下一次 `/upgrade` 會從空白的 `upgrade.txt` 重新累積。
  （若更新的是 `character_persona.json` 則不觸發，因為那跟「功能許願」無關。）

你提供的兩個舊格式歸檔檔案，我已經幫你示範轉換成新格式放在
`updates/done/done260814.txt`、`updates/done/done260816.txt`（原始檔仍保留在 `old/` 沒有動，
只是複製了一份新命名版本方便你比對）。

## ✅ 沒有變動的部分
- `old/` 資料夾與其自動升級備份腳本邏輯完全沒有動（依你指示）。
- 好感度 / RAG 記憶 / 人設對話邏輯 / 所有 slash command 的行為本身沒有變動，只調整了檔案路徑與鎖。

## 🌱 2026-08-17 追加：diana_evolution.py 整併與重寫

比對你上傳的兩份檔案：`diana_evolution.py`（無數字，補上傳）和 `diana_evolution2.py`。
兩者除了 `diana_evolution2.py` 多了「讀取 `upgrade.txt` 主動升級」功能之外，其餘邏輯完全相同——
`diana_evolution2.py` 是 `diana_evolution.py` 的超集合／新版，不是各有優劣需要取捨的情況。

**保留 `diana_evolution2.py` 的邏輯為基礎**，兩個舊檔都改名保留：
- `old_diana_evolution.py`（無數字舊版）
- `old_diana_evolution2.py`（有主動升級功能的版本）

重寫後的 `diana_evolution.py` 修正了幾個實際問題：

1. **API 金鑰檔名對不上的 bug**：舊版讀 `google API key.txt`，但 `bot.py` 讀的是
   `google API key -2.txt`——兩支程式指向不同檔案。新版統一改讀
   `config/google API key -2.txt`，並保留舊路徑的自動遷移邏輯。
2. **路徑改用新資料夾結構**：`config/` `data/` `logs/` `updates/`，跟重整後的 `bot.py` 完全對齊；
   `old/` 版本備份資料夾維持不動，兩支程式共用同一套 `{type}_{YYMM}.{seq}` 版本號邏輯，
   序號不會互相衝突。
3. **`upgrade.txt` 歸檔格式統一**：從舊的 `upgrade.txt.done_YYYYMMDD_HHMM` 改成跟 `bot.py`
   一致的 `updates/done/done{YYMMDD}.txt`。
4. **補上關鍵的重啟串接機制（原本兩個舊版都沒有這個功能）**：
   舊版腳本改完 `bot.py` 後只是把新程式碼寫到硬碟，**沒有讓正在跑的 bot 行程真的套用新版本**
   （檔案覆寫不會讓已經載入記憶體的 process 自動更新）。新版在 `evolve_code()` 產生新版本後，
   會建立 `data/pending_restart.flag`；`bot.py` 的背景維護任務（區塊 A0）每 60 秒會檢查一次，
   偵測到旗標就會私訊主人告知原因、安全重啟並清除旗標，讓演化結果真正生效。
   （若只是人設 `character_persona.json` 更新則不會觸發重啟，因為那份是執行期讀取，不需要重啟。）

**執行方式沒有變動**：預設每天凌晨 04:00 自動跑一次，也可以用
`python diana_evolution.py --run-now` 手動觸發一次。

## 🔑 2026-08-17 追加：聊天端與演化引擎金鑰完全分離

原本 `bot.py` 和 `diana_evolution.py` 共用同一份 `config/google API key -2.txt`，這造成一個實際的 bug：

- `diana_evolution.py` 寫入這個檔案是**兩行**格式（第一行金鑰、第二行模型名稱）。
- 但 `bot.py` 的 `get_google_api_key()` 用 `f.read().strip()` 讀**整個檔案**當成 api_key，
  沒有只取第一行。一旦這個共用檔案裡有第二行模型名稱，`bot.py` 抓到的金鑰字串會變成
  `"你的金鑰\n模型名稱"`（中間夾著換行），拿去打 Google API 一定認證失敗。
- 另外兩支程式各自用不同排程觸發，`diana_evolution.py` 整檔覆寫金鑰檔的當下，
  如果剛好撞上 `bot.py` 啟動/重啟去讀同一份檔案，也有讀到寫一半內容的風險。

**修法**：兩邊金鑰檔完全分開存放，互不讀寫對方的檔案：

| 程式 | 金鑰檔路徑 | 格式 |
|---|---|---|
| `bot.py`（聊天） | `config/google_api_key_chat.txt` | 單行金鑰 |
| `diana_evolution.py`（演化） | `config/google_api_key_evolution.txt` | 金鑰 + 模型名稱（兩行） |

- `bot.py` 的 `get_google_api_key()` 也順手修正成只讀第一行，就算檔案格式不小心混進第二行也不會壞掉。
- 兩邊都各自可以指定不同的 Google API 專案/方案/額度，符合你「聊天用便宜方案、演化用不同方案」的需求。
- 都有向下相容的自動遷移：如果偵測到舊的共用金鑰檔還在，`bot.py` 會自動搬過去（只取第一行）；
  `diana_evolution.py` 第一次啟動找不到自己的專屬金鑰檔時，會**複製**（不是搬移）一份舊金鑰檔當起始值，
  不會動到 `bot.py` 正在使用的檔案，你之後可以自行把兩邊金鑰換成不同的方案。

