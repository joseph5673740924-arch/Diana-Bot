"""Diana Bot Control Center — install, configure, launch and observe Diana without changing her core workflow."""
from __future__ import annotations
import json, os, sys, subprocess, threading, queue, time, urllib.request, urllib.error
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config"; DATA = ROOT / "data"
GUI_CONFIG = CONFIG / "gui_config.json"; PERSONA = CONFIG / "character_persona.json"
DIANA_EXAMPLE = CONFIG / "persona_diana_example.json"
TOKEN = CONFIG / "discord_token.txt"
CHAT_KEY = CONFIG / "google_api_key_chat.txt"
EVOLUTION_KEY = CONFIG / "google_api_key_evolution.txt"
IMAGE_KEY = CONFIG / "google_api_key_image.txt"
STATUS = DATA / "runtime_status.json"

DEFAULT_CFG = {
    "local_base_url":"http://127.0.0.1:1234/v1", "local_model":"google/gemma-4-e2b",
    "web_base_url":"https://generativelanguage.googleapis.com/v1beta/openai/", "web_model":"gemini-3.5-flash-lite",
    "image_endpoint":"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-image:generateContent",
    "image_model":"gemini-3.1-flash-image", "temperature":0.75, "thinking_tokens":256
}
STAGES = [("idle","等待"),("analyzing","分析"),("memory","記憶查詢"),("thinking","LLM 思考"),("replying","回覆中"),("memory_cleanup","整理記憶")]


def read_json(path, default):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return default.copy() if isinstance(default, dict) else default

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8"); os.replace(tmp,path)

def read_first(path):
    try: return path.read_text(encoding="utf-8").splitlines()[0].strip()
    except Exception: return ""

def write_first_preserve(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    rest=[]
    if path.exists(): rest=path.read_text(encoding="utf-8").splitlines()[1:]
    path.write_text("\n".join([value.strip()]+rest).rstrip()+"\n",encoding="utf-8")

class App(tk.Tk):
    def __init__(self):
        super().__init__(); os.chdir(ROOT); CONFIG.mkdir(exist_ok=True); DATA.mkdir(exist_ok=True)
        self.title("Diana Bot Control Center"); self.geometry("1080x720"); self.minsize(920,620)
        self.configure(bg="#111827"); self.proc=None; self.logq=queue.Queue(); self.cfg={**DEFAULT_CFG,**read_json(GUI_CONFIG,{})}
        self._style(); self._layout(); self.load_settings(); self.after(300,self.tick); self.protocol("WM_DELETE_WINDOW",self.close)
        if not GUI_CONFIG.exists(): self.after(500,self.show_welcome)
    def _style(self):
        st=ttk.Style(self); st.theme_use("clam")
        st.configure("TFrame",background="#111827"); st.configure("Card.TFrame",background="#1f2937")
        st.configure("TLabel",background="#111827",foreground="#e5e7eb",font=("Segoe UI",10)); st.configure("Title.TLabel",font=("Segoe UI",22,"bold"),foreground="white")
        st.configure("Card.TLabel",background="#1f2937",foreground="#e5e7eb"); st.configure("Head.Card.TLabel",background="#1f2937",foreground="white",font=("Segoe UI",13,"bold"))
        st.configure("TButton",font=("Segoe UI",10),padding=8); st.configure("Nav.TButton",anchor="w",padding=12)
        st.configure("TEntry",fieldbackground="#374151",foreground="white",insertcolor="white")
        st.configure("TNotebook",background="#111827",borderwidth=0); st.configure("TNotebook.Tab",padding=(14,8))
    def _layout(self):
        side=ttk.Frame(self,width=190); side.pack(side="left",fill="y",padx=(14,0),pady=14); side.pack_propagate(False)
        ttk.Label(side,text="DIANA",style="Title.TLabel").pack(anchor="w",pady=(8,22))
        self.body=ttk.Frame(self); self.body.pack(side="left",fill="both",expand=True,padx=14,pady=14)
        self.pages={}
        for name in ["首頁","連線設定","角色設定","安裝與進階"]:
            f=ttk.Frame(self.body); self.pages[name]=f
            ttk.Button(side,text=name,style="Nav.TButton",command=lambda n=name:self.show(n)).pack(fill="x",pady=3)
        self.build_home(); self.build_connections(); self.build_persona(); self.build_advanced(); self.show("首頁")
    def show(self,name):
        for f in self.pages.values(): f.pack_forget()
        self.pages[name].pack(fill="both",expand=True)
    def card(self,parent,title):
        f=ttk.Frame(parent,style="Card.TFrame",padding=18); ttk.Label(f,text=title,style="Head.Card.TLabel").pack(anchor="w",pady=(0,12)); return f
    def build_home(self):
        p=self.pages["首頁"]; ttk.Label(p,text="Diana Bot Control Center",style="Title.TLabel").pack(anchor="w")
        self.online=ttk.Label(p,text="● OFFLINE"); self.online.pack(anchor="w",pady=(4,16))
        c=self.card(p,"目前狀態"); c.pack(fill="x")
        self.stage_widgets={}; row=ttk.Frame(c,style="Card.TFrame"); row.pack(fill="x",pady=10)
        for key,label in STAGES:
            w=ttk.Label(row,text=f"○  {label}",style="Card.TLabel",font=("Segoe UI",10,"bold")); w.pack(side="left",expand=True); self.stage_widgets[key]=w
        self.detail=ttk.Label(c,text="尚未啟動",style="Card.TLabel"); self.detail.pack(anchor="w",pady=(8,0))
        b=ttk.Frame(p); b.pack(fill="x",pady=14); ttk.Button(b,text="▶ 啟動 Diana",command=self.start_bot).pack(side="left",padx=(0,8)); ttk.Button(b,text="■ 停止",command=self.stop_bot).pack(side="left",padx=8); ttk.Button(b,text="↻ 重新啟動",command=self.restart_bot).pack(side="left",padx=8)
        lc=self.card(p,"執行紀錄"); lc.pack(fill="both",expand=True); self.log=scrolledtext.ScrolledText(lc,height=16,bg="#0b1220",fg="#d1d5db",insertbackground="white",font=("Consolas",9),relief="flat"); self.log.pack(fill="both",expand=True)
    def field(self,parent,label,var,show=None):
        r=ttk.Frame(parent,style="Card.TFrame"); r.pack(fill="x",pady=4); ttk.Label(r,text=label,style="Card.TLabel",width=20).pack(side="left"); e=ttk.Entry(r,textvariable=var,show=show); e.pack(side="left",fill="x",expand=True); return e
    def build_connections(self):
        p=self.pages["連線設定"]; ttk.Label(p,text="連線設定",style="Title.TLabel").pack(anchor="w",pady=(0,12))
        self.vars={k:tk.StringVar() for k in ["local_base_url","local_model","web_base_url","web_model","image_endpoint","image_model","discord","chat_key","evolution_key","image_key"]}
        nb=ttk.Notebook(p); nb.pack(fill="both",expand=True)
        for title in ["本地 API","Web API","Image API","Discord / Keys"]: nb.add(ttk.Frame(nb),text=title)
        tabs=nb.winfo_children()
        c=self.card(tabs[0],"Local LLM（維持原本本地 API 流程）"); c.pack(fill="x",padx=12,pady=12); self.field(c,"Base URL",self.vars["local_base_url"]); self.field(c,"Model",self.vars["local_model"]); ttk.Button(c,text="測試本地 API",command=self.test_local).pack(anchor="e",pady=8)
        c=self.card(tabs[1],"Web LLM"); c.pack(fill="x",padx=12,pady=12); self.field(c,"Base URL",self.vars["web_base_url"]); self.field(c,"Model",self.vars["web_model"]); self.field(c,"Chat API Key",self.vars["chat_key"],"•"); ttk.Button(c,text="測試 Web API",command=self.test_web).pack(anchor="e",pady=8)
        c=self.card(tabs[2],"Image API"); c.pack(fill="x",padx=12,pady=12); self.field(c,"Endpoint",self.vars["image_endpoint"]); self.field(c,"Model",self.vars["image_model"]); self.field(c,"Image API Key",self.vars["image_key"],"•"); ttk.Button(c,text="檢查設定",command=lambda:self.info("Image API 設定已填寫。實際生圖仍使用原本 image_gen.py 流程。" )).pack(anchor="e",pady=8)
        c=self.card(tabs[3],"獨立憑證檔案"); c.pack(fill="x",padx=12,pady=12); self.field(c,"Discord Token",self.vars["discord"],"•"); self.field(c,"Chat API Key",self.vars["chat_key"],"•"); self.field(c,"Evolution API Key",self.vars["evolution_key"],"•"); self.field(c,"Image API Key",self.vars["image_key"],"•"); ttk.Label(c,text="三個 Google/API Key 維持 chat / evolution / image 三個獨立檔案；GUI 不合併它們。",style="Card.TLabel").pack(anchor="w",pady=8)
        ttk.Button(p,text="儲存連線設定",command=self.save_settings).pack(anchor="e",pady=10)
    def build_persona(self):
        p=self.pages["角色設定"]; ttk.Label(p,text="角色設定",style="Title.TLabel").pack(anchor="w")
        b=ttk.Frame(p); b.pack(fill="x",pady=10); ttk.Button(b,text="載入黛安娜範例",command=self.load_diana).pack(side="left",padx=(0,8)); ttk.Button(b,text="空白規範模板",command=self.blank_persona).pack(side="left",padx=8); ttk.Button(b,text="✨ LLM 生成三案",command=self.generate_personas).pack(side="left",padx=8); ttk.Button(b,text="儲存 Persona",command=self.save_persona).pack(side="right")
        self.persona_prompt=tk.StringVar(); ttk.Entry(p,textvariable=self.persona_prompt).pack(fill="x",pady=(0,8)); ttk.Label(p,text="上方可用自然語言描述角色；LLM 生成時會產生三個候選方案。 ").pack(anchor="w")
        self.persona_text=scrolledtext.ScrolledText(p,bg="#0b1220",fg="#e5e7eb",insertbackground="white",font=("Consolas",10),relief="flat"); self.persona_text.pack(fill="both",expand=True,pady=10)
    def build_advanced(self):
        p=self.pages["安裝與進階"]; ttk.Label(p,text="安裝與進階",style="Title.TLabel").pack(anchor="w",pady=(0,12))
        c=self.card(p,"安裝 / 檢查"); c.pack(fill="x"); ttk.Button(c,text="安裝 requirements.txt",command=self.install_deps).pack(side="left",padx=(0,8)); ttk.Button(c,text="檢查必要檔案",command=self.check_files).pack(side="left",padx=8); ttk.Button(c,text="開啟專案資料夾",command=self.open_folder).pack(side="left",padx=8)
        c2=self.card(p,"Bot 參數"); c2.pack(fill="x",pady=12); self.temp=tk.StringVar(); self.think=tk.StringVar(); self.field(c2,"Temperature",self.temp); self.field(c2,"Thinking tokens",self.think); ttk.Button(c2,text="儲存",command=self.save_settings).pack(anchor="e",pady=8)
        ttk.Label(p,text="GUI 只管理安裝、設定、啟停與狀態；Bot 對話、記憶、Discord、Evolution 等既有流程不在這版重構。",wraplength=800).pack(anchor="w",pady=8)
    def load_settings(self):
        for k in ["local_base_url","local_model","web_base_url","web_model","image_endpoint","image_model"]: self.vars[k].set(str(self.cfg.get(k,DEFAULT_CFG[k])))
        self.vars["discord"].set(read_first(TOKEN)); self.vars["chat_key"].set(read_first(CHAT_KEY)); self.vars["evolution_key"].set(read_first(EVOLUTION_KEY)); self.vars["image_key"].set(read_first(IMAGE_KEY))
        self.temp.set(str(self.cfg.get("temperature",.75))); self.think.set(str(self.cfg.get("thinking_tokens",256))); self.load_diana()
    def save_settings(self):
        try:
            for k in ["local_base_url","local_model","web_base_url","web_model","image_endpoint","image_model"]: self.cfg[k]=self.vars[k].get().strip()
            self.cfg["temperature"]=float(self.temp.get()); self.cfg["thinking_tokens"]=int(self.think.get()); write_json(GUI_CONFIG,self.cfg)
            write_first_preserve(TOKEN,self.vars["discord"].get()); write_first_preserve(CHAT_KEY,self.vars["chat_key"].get()); write_first_preserve(EVOLUTION_KEY,self.vars["evolution_key"].get()); write_first_preserve(IMAGE_KEY,self.vars["image_key"].get())
            self.info("設定已儲存。Bot 若正在執行，連線設定會在下次重啟後套用。")
        except Exception as e: messagebox.showerror("儲存失敗",str(e))
    def load_diana(self):
        obj=read_json(DIANA_EXAMPLE,read_json(PERSONA,{"name":"黛安娜"})); self.persona_text.delete("1.0","end"); self.persona_text.insert("1.0",json.dumps(obj,ensure_ascii=False,indent=2))
    def blank_persona(self):
        obj={"name":"","age_style":"","background":"","relationship":"","guest_attitude":"","personality_traits":[""],"speech_patterns":[""],"tone_rules":[""],"few_shot_examples":[{"user":"","diana":""}],"favorability_reactions":{"low":"","medium":"","high":""}}
        self.persona_text.delete("1.0","end"); self.persona_text.insert("1.0",json.dumps(obj,ensure_ascii=False,indent=2))
    def save_persona(self):
        try: write_json(PERSONA,json.loads(self.persona_text.get("1.0","end"))); self.info("Persona 已儲存。")
        except Exception as e: messagebox.showerror("Persona JSON 無效",str(e))
    def generate_personas(self):
        desc=self.persona_prompt.get().strip()
        if not desc: return messagebox.showwarning("需要描述","先在上方輸入你想要的角色描述。")
        self.info("將使用目前設定的 LLM 產生三個角色方案；完成後會讓你選擇。")
        threading.Thread(target=self._gen_personas,args=(desc,),daemon=True).start()
    def _gen_personas(self,desc):
        schema='name, age_style, background, relationship, guest_attitude, personality_traits(array), speech_patterns(array), tone_rules(array), few_shot_examples(array of {user,diana}), favorability_reactions({low,medium,high})'
        prompt=f'依描述建立三個差異明顯但都可直接使用的角色 Persona。描述：{desc}\n欄位必須是：{schema}\n只輸出 JSON array，恰好三個 object，不要 markdown。'
        try:
            # Prefer local API; no new backend abstraction is introduced.
            base=self.vars["local_base_url"].get().rstrip("/"); model=self.vars["local_model"].get(); url=base+"/chat/completions"
            payload=json.dumps({"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0.9}).encode()
            req=urllib.request.Request(url,data=payload,headers={"Content-Type":"application/json","Authorization":"Bearer lm-studio"})
            with urllib.request.urlopen(req,timeout=120) as r: data=json.loads(r.read().decode())
            text=data["choices"][0]["message"]["content"].strip(); text=text.removeprefix("```json").removesuffix("```").strip(); arr=json.loads(text)
            self.after(0,lambda:self.choose_persona(arr))
        except Exception as e: self.after(0,lambda:messagebox.showerror("LLM 生成失敗",str(e)))
    def choose_persona(self,arr):
        if not isinstance(arr,list) or len(arr)<3: return messagebox.showerror("格式錯誤","LLM 沒有回傳三個 Persona。")
        w=tk.Toplevel(self); w.title("選擇 Persona"); w.geometry("900x600"); nb=ttk.Notebook(w); nb.pack(fill="both",expand=True,padx=10,pady=10)
        for i,obj in enumerate(arr[:3]):
            f=ttk.Frame(nb); nb.add(f,text=f"方案 {chr(65+i)} · {obj.get('name','未命名')}"); t=scrolledtext.ScrolledText(f,bg="#0b1220",fg="white",font=("Consolas",9)); t.pack(fill="both",expand=True); t.insert("1.0",json.dumps(obj,ensure_ascii=False,indent=2)); t.configure(state="disabled")
        def use():
            obj=arr[nb.index(nb.select())]; self.persona_text.delete("1.0","end"); self.persona_text.insert("1.0",json.dumps(obj,ensure_ascii=False,indent=2)); w.destroy()
        ttk.Button(w,text="套用目前方案",command=use).pack(pady=(0,10))
    def start_bot(self):
        if self.proc and self.proc.poll() is None: return self.info("Diana 已在執行。")
        self.save_settings(); self.log.insert("end","\n=== Starting Diana ===\n"); self.proc=subprocess.Popen([sys.executable,"-u","bot.py"],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1)
        threading.Thread(target=self._pump,daemon=True).start()
    def _pump(self):
        if self.proc and self.proc.stdout:
            for line in self.proc.stdout: self.logq.put(line)
    def stop_bot(self):
        if self.proc and self.proc.poll() is None: self.proc.terminate(); self.logq.put("=== Diana stopped ===\n")
    def restart_bot(self): self.stop_bot(); self.after(900,self.start_bot)
    def tick(self):
        try:
            while True: self.log.insert("end",self.logq.get_nowait()); self.log.see("end")
        except queue.Empty: pass
        running=bool(self.proc and self.proc.poll() is None); self.online.config(text="● ONLINE" if running else "● OFFLINE")
        st=read_json(STATUS,{}) if STATUS.exists() else {}; stage=st.get("stage","") if running else ""
        for k,w in self.stage_widgets.items(): w.config(text=("●  " if k==stage else "○  ")+dict(STAGES)[k])
        if running: self.detail.config(text=st.get("detail","正在啟動…"))
        self.after(350,self.tick)
    def test_local(self): self._test_openai(self.vars["local_base_url"].get(),None,"本地 API")
    def test_web(self): self._test_openai(self.vars["web_base_url"].get(),self.vars["chat_key"].get(),"Web API")
    def _test_openai(self,base,key,label):
        def run():
            try:
                req=urllib.request.Request(base.rstrip("/")+"/models",headers={"Authorization":"Bearer "+(key or "lm-studio")}); urllib.request.urlopen(req,timeout=8).read(); self.after(0,lambda:self.info(label+" 連線成功。"))
            except Exception as e:self.after(0,lambda:messagebox.showerror(label+" 連線失敗",str(e)))
        threading.Thread(target=run,daemon=True).start()
    def install_deps(self):
        def run():
            self.logq.put("=== Installing requirements ===\n"); p=subprocess.Popen([sys.executable,"-m","pip","install","-r","requirements.txt"],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace");
            for line in p.stdout:self.logq.put(line)
            p.wait(); self.logq.put(f"=== pip finished: {p.returncode} ===\n")
        threading.Thread(target=run,daemon=True).start(); self.show("首頁")
    def check_files(self):
        items={"bot.py":ROOT/"bot.py","Persona":PERSONA,"Discord Token":TOKEN,"Chat Key":CHAT_KEY,"Evolution Key":EVOLUTION_KEY,"Image Key":IMAGE_KEY,"Reference images":ROOT/"img"/"reference"}
        messagebox.showinfo("必要檔案","\n".join(("✓ " if p.exists() else "✗ ")+n for n,p in items.items()))
    def open_folder(self):
        if sys.platform.startswith("win"): os.startfile(ROOT)
        elif sys.platform=="darwin": subprocess.Popen(["open",ROOT])
        else: subprocess.Popen(["xdg-open",ROOT])
    def show_welcome(self):
        if messagebox.askyesno("歡迎使用 Diana","第一次使用？GUI 可以協助你完成 API、Discord、角色與安裝設定。\n\n現在前往連線設定嗎？"): self.show("連線設定")
    def info(self,msg): messagebox.showinfo("Diana",msg)
    def close(self):
        if self.proc and self.proc.poll() is None and messagebox.askyesno("關閉","Diana 還在執行。要一起停止 Bot 嗎？"): self.stop_bot()
        self.destroy()

if __name__=="__main__": App().mainloop()
