# Diana Bot

Diana 是一個以 Python 建構的 **Persistent AI Companion / AI Agent 原型**。

它並不是單純將 Discord 訊息轉送給 LLM 的聊天機器人。Diana 本身會持續運行並維護自己的記憶、角色設定、使用者關係與系統狀態；LLM 則作為需要理解、推理與生成內容時使用的智能模組。

目前 Discord 是 Diana 的主要互動介面。
## 📖 Project Documents

如果你對 Diana 為什麼誕生，以及這個專案背後的想法有興趣：

- [開發故事：從 Neuro-sama 到 Diana 的八年]
- [Diana AI Agent 開發與效能優化日誌]

---

## ✨ 目前功能

### 💬 AI 對話

Diana 可以透過 Discord 與使用者進行自然語言對話。

系統會根據：

* 角色 Persona
* 最近對話
* 長期記憶
* 使用者資料
* 關係狀態
* 當前系統狀態

建立提供給 LLM 的上下文，而不是每次都進行完全獨立的聊天。

---

## 🧠 長期與短期記憶

Diana 具有持續性的記憶系統。

### Short-term Memory

近期對話與重要上下文會保存在本地資料中，用於維持目前對話的連續性。

### Long-term Memory

系統使用 ChromaDB 建立長期記憶資料庫。

需要時 Diana 可以搜尋與目前話題相關的過去記憶，再將結果提供給 LLM。

因此重新啟動 Bot 並不代表 Diana 完全失去之前的記憶。

---

## 👤 使用者與關係狀態

Diana 不只保存聊天紀錄，也可以維護與使用者相關的狀態，例如：

* 使用者資料
* 關係資訊
* 好感度
* 互動狀態
* 長期記憶
* 角色自身狀態

這些資料會影響後續互動，而不是只存在單次 Prompt 中。

---

## 💤 主動行為

Diana 並不一定需要等待使用者先傳送訊息。

系統具有自己的持續運行邏輯，可以根據目前狀態進行主動行為，例如：

* 主動發起對話
* 執行提醒
* 根據無聊值等狀態決定是否互動
* 執行週期性背景工作
* 整理記憶

因此 Diana 的 Python Runtime 才是持續存在的主體，LLM 並不需要一直保持運行。

---

## ⏰ Reminder

Diana 具有提醒功能。

提醒會由系統保存並在指定條件或時間觸發，而不是依賴 LLM 一直記住提醒內容。

---

## 🎭 Persona 系統

角色人格由獨立 Persona 設定管理。

使用者可以：

* 使用內建的 Diana Persona
* 自己建立角色
* 修改角色背景與個性
* 設定角色與使用者的關係
* 設定說話方式與行為規範
* 使用 LLM 協助生成角色設定

GUI 提供三種 Persona 建立方式：

**Diana Example**

直接使用 Diana 的預設角色設定。

**Blank Template**

從空白角色模板建立自己的 AI 角色。

**AI Generate**

輸入自然語言描述後，由 LLM 產生三個不同的 Persona 方案供使用者選擇與修改。

---

## 🖼️ 圖片功能

Diana 具有獨立的 Image API。

圖片系統可以使用角色 Reference Images，在產生圖片時盡可能維持角色外觀與風格一致。

系統也可以依照 Diana 當下的動作或情境使用對應圖片。

如果缺少需要的角色動作圖片，可以透過 Image API 建立新的圖片。

---

## 🌐 三套獨立 AI API

Diana 目前使用三套彼此獨立的 AI API：

### Local API

主要的本地 LLM API。

可搭配提供 OpenAI-compatible API 的本地推理服務使用。

例如：

`http://127.0.0.1:1234/v1`

適合使用 LM Studio 等本地 LLM Server。

### Web API

獨立的網路 AI API。

用於需要另一套線上模型能力的功能，不與 Local API 共用設定。

### Image API

獨立的圖片生成 API。

負責 Diana 的圖片相關功能。

三套 API 的 Key **彼此完全分離**，不共用同一個 Key 設定檔。

---

# 🖥️ Diana Control Center

Diana 現在提供圖形化控制介面，不需要手動修改所有設定檔才能啟動。

啟動：

`start_gui.bat`

即可開啟 Diana Control Center。

---

## 🚀 安裝與初始化

GUI 可以協助完成：

* Dependencies 安裝
* Local API 設定
* Web API 設定
* Image API 設定
* Discord Token 設定
* Persona 設定
* Bot 啟動

API Key 與 Discord Token 會分別保存，避免不同 API Credential 互相干擾。

---

## 🔌 API Configuration

GUI 可以設定 Diana 使用的：

**Local API**

* Endpoint
* Model
* API Key

**Web API**

* Endpoint
* Model
* API Key

**Image API**

* Endpoint
* Model
* API Key

三個 API 的設定與 Key 均獨立管理。

---

## 🎮 Bot Control

完成設定後，可以直接從 GUI：

* Start
* Stop
* Restart

Diana。

因此日常使用不需要再手動開啟 Terminal 執行 Python 指令。

---

# 📊 Runtime Status

Control Center 可以顯示 Diana 目前正在進行的主要工作階段。

例如：

`等待 → 分析 → 記憶查詢 → LLM 思考 → 回覆 → 整理記憶 → 等待`

GUI 會亮起目前所在的階段，讓使用者可以直接看到 Diana 現在正在做什麼。

這些狀態只是在原有 Bot 流程中加入 GUI 顯示，不會改變原本的 Bot 工作方式。

---

# 🧬 Evolution

Diana 包含獨立的 Evolution 系統。

Evolution 可以定期分析目前的角色與系統狀態，並進行成長或調整。

Evolution 與主要 Discord Bot 分開運行，具有自己的：

* API
* Model
* Log
* Backup
* Version
* Restart 機制

系統在修改重要內容前會建立備份，並記錄成長過程。

Evolution 的目標不是單純讓 LLM 每次產生不同回答，而是讓部分變化能夠真正保存在 Diana 的持久系統中。

---

# 🏗️ 基本運作方式

Diana 的基本互動流程可以簡化為：

```text
Discord Message
      ↓
Python Runtime
      ↓
Analyze
      ↓
Memory Retrieval
      ↓
Persona + Context
      ↓
LLM
      ↓
Response
      ↓
Memory / State Update
      ↓
Discord
```

LLM 是 Diana 的推理與語言能力之一，而不是整個 Diana Runtime。

即使沒有正在呼叫 LLM，Python 系統仍然可以維護：

* 狀態
* 記憶
* Reminder
* 排程
* 主動行為
* 資料保存

---

# 📦 如何使用

### 1. 下載專案

下載 Release 或 Clone Repository。

### 2. 啟動 GUI

Windows 執行：

```text
start_gui.bat
```

### 3. 安裝 Dependencies

在 GUI 中執行依賴套件安裝。

### 4. 設定 AI API

至少確認需要使用的 API 已正確設定。

Local API、Web API 與 Image API 為三套獨立設定。

### 5. 設定 Discord

輸入自己的 Discord Bot Token。

Discord Bot 必須先在 Discord Developer Portal 建立。

### 6. 建立角色

選擇：

* Diana Example
* Blank Template
* AI Generate

其中一種方式建立 Persona。

### 7. 啟動 Diana

回到主頁按下 Start。

連線成功後即可從 Discord 與 Diana 互動。

---

# ⚠️ Project Status

Diana 目前仍屬於 **Early Prototype / Experimental AI Agent**。

目前版本的重點不是建立另一個通用聊天 UI，而是實驗：

> 如果將 LLM 當成「需要思考時才呼叫的智能模組」，並把記憶、狀態、關係、排程與行為保存在持續運行的軟體系統中，能否逐漸建立一個具有長期連續性的 AI Companion？

因此目前程式仍可能存在 Bug、未完成介面或實驗性功能。

不建議直接用於需要高可靠度或安全關鍵的用途。

---

# 🛣️ Future Direction

Diana 未來的方向是逐步從 Discord Companion 擴展成更加完整的 Persistent AI Agent。

預計研究方向包括：

* 更完整的 World Model
* Policy / Permission System
* 更自然的長期記憶
* System 1 / System 2 分工
* Vision
* Audio / Voice
* Web Tools
* Smart Home
* 更多 Interface Adapter
* 更成熟的自主行為
* 從經驗中形成新的持久能力

Discord 最終只是 Diana 與外部世界互動的其中一種介面。

---

## 核心理念

**Model Intelligence ≠ System Intelligence**

一個更強的 LLM 不一定等於一個更完整的 AI Agent。

Diana 的實驗方向是：

**LLM 負責理解、推理與生成；Python 系統負責記住、執行、維持狀態與持續存在。**

讓一次性的推理結果有機會逐漸成為可以被系統保存、再次使用，甚至轉化成持久能力的經驗。
