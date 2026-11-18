# ═══ 多策略 RAG 文件問答（Streamlit 線上版）· Groq + ChromaDB + LINE Bot 投放 ═══
# 執行：streamlit run app.py
from __future__ import annotations

import csv
import hmac
import io
import os
import re
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import chromadb
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import streamlit as st
from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.shared import Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph
from groq import Groq
from linebot.v3.messaging import ApiClient, BroadcastRequest, Configuration, MessagingApi, TextMessage
from linebot.v3.messaging.exceptions import ApiException
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


def secret(name: str, default: str = "") -> str:
    """優先讀取 st.secrets（.streamlit/secrets.toml），其次讀環境變數。"""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:  # 沒有 secrets.toml 時可能拋出例外
        pass
    return os.getenv(name, default).strip()


# ── 設定 ──
EMBED_MODEL = secret("EMBED_MODEL", "intfloat/multilingual-e5-small")
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
CHUNK, OVERLAP, CHILD, CHILD_OVERLAP, MAX_CTX = 500, 100, 200, 40, 7000
ENV_KEY = secret("GROQ_API_KEY")
ENV_LINE = secret("LINE_CHANNEL_ACCESS_TOKEN")
ADMIN_PASSWORD = secret("ADMIN_PASSWORD")           # 未設定則整個 LINE 投放區塊隱藏
MAX_UPLOAD_MB = int(secret("MAX_UPLOAD_MB", "20"))
DOC_TTL = int(secret("DOC_TTL_MIN", "60")) * 60     # 文件索引閒置多久後釋放（秒）
LINE_LIMIT = int(secret("LINE_LIMIT", "300"))       # LINE 投放字數上限（可在 Secrets 調整）
LINE_HOST = "https://api.line.me"
EXAMPLES = ("這份文件的重點是什麼？", "文件提到哪些規定或限制？", "有哪些風險或注意事項？")

STRATEGIES = {
    "hybrid": ("⚡ 混合搜尋", "語意 + 關鍵字，RRF 融合排名（推薦）"),
    "semantic": ("🔍 語意搜尋", "向量相似度檢索"),
    "tfidf": ("📊 關鍵字", "字元 n-gram，適合專有名詞、條號、數字"),
    "rerank": ("🎯 重新排序", "LLM 對候選片段批次評分後重排"),
    "multi_query": ("🔄 多查詢", "LLM 改寫 3 個角度問題聯合檢索"),
    "compress": ("✂️ 壓縮", "LLM 抽取每段最相關的句子"),
    "parent_child": ("📂 父子文檔", "小片段定位，回傳完整大片段"),
    "hyde": ("💡 HyDE", "先生成假設答案再檢索"),
}
_SENT = re.compile(r"(?<=[。！？!?；;])|\n+")


def split_text(text, size, overlap):
    sents = [s.strip() for s in _SENT.split(re.sub(r"[ \t　\xa0]+", " ", text)) if s and s.strip()]
    chunks, buf, fresh = [], "", 0
    for p in (s[i:i + size] for s in sents for i in range(0, len(s), size)):
        if fresh and len(buf) + len(p) + 1 > size:
            chunks.append(buf)
            buf, fresh = (buf[-overlap:].lstrip() if overlap else ""), 0
        buf = f"{buf} {p}" if buf else p
        fresh += len(p)
    return chunks + ([buf.strip()] if fresh else [])


@dataclass
class Doc:
    sid: str
    name: str
    chunks: list
    labels: list
    n_child: int = 0
    tfidf: object = field(default=None, repr=False)
    tm: object = field(default=None, repr=False)


class LLM:
    def __init__(self, key, model):
        key = (key or "").strip()
        self.client = Groq(api_key=key) if key else None
        self.model = model or MODELS[0]

    def __call__(self, prompt, max_tokens=300, temp=0.3, system=None, raise_err=False):
        if not self.client:
            return None
        kw = dict(model=self.model, temperature=temp, max_completion_tokens=max_tokens,
                  messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}])
        if "gpt-oss" in self.model:  # 推理模型：推理 token 也計入上限
            kw.update(max_completion_tokens=max(max_tokens * 3, 1024), reasoning_effort="low", include_reasoning=False)
        try:
            try:
                r = self.client.chat.completions.create(**kw)
            except Exception as e:
                if "reasoning" not in str(e).lower():
                    raise
                kw.pop("reasoning_effort", None)
                kw.pop("include_reasoning", None)
                r = self.client.chat.completions.create(**kw)
            return re.sub(r"<think>.*?</think>", "", r.choices[0].message.content or "", flags=re.S).strip()
        except Exception:
            if raise_err:
                raise
            return None


class KeyVault:
    """以 AES-256-GCM 加密保存使用者輸入的 API Key。
    加密金鑰為每個工作階段隨機產生，只存在記憶體中，不寫入磁碟；
    每次加密使用新的隨機 nonce，並以 AAD 綁定用途，防止密文被挪作他用。"""

    _AAD = b"groq-api-key/v1"

    def __init__(self):
        self._aes = AESGCM(AESGCM.generate_key(bit_length=256))
        self._blob: bytes | None = None

    def put(self, plain: str) -> None:
        nonce = secrets.token_bytes(12)
        self._blob = nonce + self._aes.encrypt(nonce, plain.encode("utf-8"), self._AAD)

    def get(self) -> str:
        if not self._blob:
            return ""
        nonce, ct = self._blob[:12], self._blob[12:]
        return self._aes.decrypt(nonce, ct, self._AAD).decode("utf-8")

    def clear(self) -> None:
        self._blob = None

    @property
    def has_key(self) -> bool:
        return self._blob is not None


def list_models(key):
    try:
        ids = [m.id for m in Groq(api_key=key.strip()).models.list().data if getattr(m, "active", True) is not False]
    except Exception:
        return []
    return sorted(i for i in ids if not any(s in i.lower() for s in ("whisper", "tts", "orpheus", "guard", "playai")))


class RAG:
    """共用資源：嵌入模型 + 向量庫。文件狀態以 Doc 物件由各使用者的 session 持有，
    不存放在這個共用物件上，避免多人使用時互相覆蓋。"""

    def __init__(self):
        self._embedder, self._lock = None, threading.Lock()
        self._db_lock = threading.Lock()
        self.e5 = "e5" in EMBED_MODEL.lower()
        self.db = chromadb.EphemeralClient()  # 記憶體模式，程式重啟即清空
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", EMBED_MODEL.split("/")[-1])[:40].strip("_")
        self.col = self.db.get_or_create_collection(f"p_{slug}", metadata={"hnsw:space": "cosine"})
        self.ccol = self.db.get_or_create_collection(f"c_{slug}", metadata={"hnsw:space": "cosine"})
        self._alive: dict[str, float] = {}   # sid -> 最後使用時間

    def embedder(self):
        if self._embedder is None:
            with self._lock:
                if self._embedder is None:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    m = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
                    m.max_seq_length = min(512, m.max_seq_length or 512)
                    self._embedder = m
        return self._embedder

    # ── 索引生命週期 ──
    def is_alive(self, sid):
        return sid in self._alive

    def release(self, sid):
        with self._db_lock:
            for c in (self.col, self.ccol):
                try:
                    c.delete(where={"sid": sid})
                except Exception:
                    pass
        self._alive.pop(sid, None)

    def purge_expired(self):
        now = time.time()
        for sid, t in list(self._alive.items()):
            if now - t > DOC_TTL:
                self.release(sid)

    # ── 文件載入 ──
    def load(self, name, data: bytes):
        t0 = time.time()
        suffix = Path(name).suffix.lower()
        if suffix not in (".pdf", ".docx"):
            raise ValueError("僅支援 PDF / DOCX")
        src = io.BytesIO(data)
        segs = self._pdf(src) if suffix == ".pdf" else self._docx(src)
        pairs = [(c, lab) for lab, txt in segs for c in split_text(txt, CHUNK, OVERLAP)]
        if not pairs:
            raise ValueError("無可擷取文字（可能是掃描檔，需先 OCR）")
        doc = Doc(uuid.uuid4().hex, name, [p[0] for p in pairs], [p[1] for p in pairs])
        self.purge_expired()
        try:
            self._add(self.col, [f"{doc.sid}_{i}" for i in range(len(pairs))], doc.chunks,
                      [{"sid": doc.sid, "idx": i} for i in range(len(pairs))])
            kids = [(f"{doc.sid}_{p}_{c}", t, p) for p, par in enumerate(doc.chunks)
                    for c, t in enumerate(split_text(par, CHILD, CHILD_OVERLAP))]
            doc.n_child = len(kids)
            self._add(self.ccol, [k[0] for k in kids], [k[1] for k in kids],
                      [{"sid": doc.sid, "idx": k[2]} for k in kids])
            doc.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, max_features=50000)
            doc.tm = doc.tfidf.fit_transform(doc.chunks)
        except Exception:
            self.release(doc.sid)
            raise
        self._alive[doc.sid] = time.time()
        return doc, f"✓ {doc.name}\n{len(doc.chunks)} 片段 · {time.time() - t0:.1f}s"

    def _pdf(self, src):
        return [(f"第 {i} 頁", t) for i, p in enumerate(PdfReader(src).pages, 1) if (t := p.extract_text() or "").strip()]

    def _docx(self, src):
        doc, segs, head, buf, nt = Document(src), [], "前言", [], 0
        flush = lambda: (segs.append((head, "\n".join(buf))), buf.clear()) if buf else None
        for el in doc.element.body.iterchildren():
            if isinstance(el, CT_P):
                para = Paragraph(el, doc)
                txt = para.text.strip()
                if not txt:
                    continue
                style = (para.style.name if para.style is not None else "") or ""
                if style.lower().startswith(("heading", "title")) or style.startswith(("標題", "标题")):
                    flush()
                    head = txt[:40]
                buf.append(txt)
            elif isinstance(el, CT_Tbl):
                nt += 1
                rows = []
                for row in Table(el, doc).rows:
                    cells = []
                    for c in row.cells:
                        t = re.sub(r"\s+", " ", c.text).strip()
                        if t and (not cells or cells[-1] != t):
                            cells.append(t)
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    flush()
                    segs.append((f"表格 {nt}（{head}）", "\n".join(rows)))
        flush()
        return segs

    # ── 向量與檢索 ──
    def _enc(self, texts, query=False):
        if self.e5:
            texts = [("query: " if query else "passage: ") + t for t in texts]
        return self.embedder().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False).tolist()

    def _add(self, col, ids, docs, metas, b=500):
        for i in range(0, len(docs), b):
            emb = self._enc(docs[i:i + b])          # 嵌入計算不持鎖，避免阻塞其他使用者
            with self._db_lock:
                col.add(ids=ids[i:i + b], documents=docs[i:i + b], metadatas=metas[i:i + b], embeddings=emb)

    def vec(self, doc, qs, k, child=False):
        total = doc.n_child if child else len(doc.chunks)
        if not total:
            return [[] for _ in qs]
        emb = self._enc(qs, True)
        with self._db_lock:
            res = (self.ccol if child else self.col).query(query_embeddings=emb, n_results=max(1, min(k, total)),
                                                           where={"sid": doc.sid}, include=["metadatas"])
        return [[int(m["idx"]) for m in ms] for ms in res["metadatas"]]

    @staticmethod
    def kw(doc, q, k):
        s = (doc.tm @ doc.tfidf.transform([q]).T).toarray().ravel()
        return [int(i) for i in np.argsort(-s)[:k] if s[i] > 0]

    @staticmethod
    def rrf(lists, k):
        sc = {}
        for lst in lists:
            for r, i in enumerate(dict.fromkeys(lst)):
                sc[i] = sc.get(i, 0) + 1 / (61 + r)
        return sorted(sc, key=sc.get, reverse=True)[:k]

    def retrieve(self, doc, q, strategy, k, llm):  # 回傳 [(片段索引, 文字)]
        hybrid = lambda n: self.rrf([self.vec(doc, [q], n * 2)[0], self.kw(doc, q, n * 2)], n)
        if strategy == "semantic":
            idx = self.vec(doc, [q], k)[0]
        elif strategy == "tfidf":
            idx = self.kw(doc, q, k)
        elif strategy == "parent_child":
            idx = list(dict.fromkeys(self.vec(doc, [q], k * 4, True)[0]))[:k]
        elif strategy == "multi_query":
            r = llm(f"將以下問題改寫成 3 個角度不同的繁體中文問題，每行一題，不加編號：\n{q}", 200, 0.7) or ""
            extra = [e for e in (l.strip("-•*·0123456789.、)） ") for l in r.splitlines()) if len(e) > 3][:3]
            idx = self.rrf(self.vec(doc, [q] + extra, k * 2), k)
        elif strategy == "hyde":
            h = llm(f"請針對以下問題寫一段約 100 字、語氣像文件原文的假設性答案（繁體中文）：\n{q}", 250, 0.7)
            idx = self.rrf(self.vec(doc, [q, h] if h else [q], k * 2), k)
        elif strategy == "rerank":
            cands = hybrid(min(max(k * 3, 6), 12))
            idx = cands[:k]
            if llm.client and len(cands) > 1:
                lst = "\n\n".join(f"[{n}] {doc.chunks[i][:400]}" for n, i in enumerate(cands, 1))
                r = llm(f"問題：{q}\n\n候選片段：\n{lst}\n\n評估每段與問題的相關度（0-10），只輸出「編號:分數」每行一個。",
                        15 * len(cands), 0) or ""
                sc = {int(a): float(b) for a, b in re.findall(r"\[?(\d+)\]?\s*[:：]\s*(\d+(?:\.\d+)?)", r)}
                idx = [cands[n] for n in sorted(range(len(cands)), key=lambda n: (-sc.get(n + 1, -1), n))[:k]]
        else:  # hybrid、compress
            idx = hybrid(k)
        hits = [(i, doc.chunks[i]) for i in idx]
        if strategy == "compress" and llm.client and hits:
            def comp(h):
                r = llm(f"從以下文本逐字擷取與問題「{q}」最相關的 1-3 句，不改寫不解釋；完全無關請輸出「無」。\n\n{h[1]}", 250, 0)
                return (h[0], r if r and r != "無" else h[1][:300])
            with ThreadPoolExecutor(min(4, len(hits))) as ex:
                hits = list(ex.map(comp, hits))
        return hits

    def answer(self, doc, q, strategy, k, model, key):
        """回傳 (回答, 參考片段, 耗時說明)。"""
        self.purge_expired()
        if doc is None or not self.is_alive(doc.sid):
            return "文件索引已過期，請重新上傳文件。", "", ""
        self._alive[doc.sid] = time.time()
        llm = LLM(key, model)
        t0 = time.time()
        hits = self.retrieve(doc, q, strategy, int(k), llm)
        t1 = time.time()
        blocks, used = [], 0
        for n, (i, txt) in enumerate(hits, 1):
            b = f"[片段{n}｜{doc.labels[i]}]\n{txt}"
            if used and used + len(b) > MAX_CTX:
                break
            blocks.append(b)
            used += len(b)
        ctx = "\n\n".join(blocks)
        src = f"📄 {doc.name} · {STRATEGIES[strategy][0]} · {len(blocks)} 片段 · 檢索 {t1 - t0:.2f}s\n{'─' * 40}\n\n{ctx}"
        if not hits:
            return "找不到相關片段，請換個問法或策略。", src, ""
        if not llm.client:
            return "⚠ 尚未設定 Groq API Key（請在側欄輸入，或由管理員設定）", src, ""
        ans = llm(f"請只根據上下文回答。若上下文沒有相關資訊，請說「文件中未提及」，不要編造。\n"
                  f"每個重點後以 [片段n] 標註依據，使用繁體中文條列整理。\n\n上下文：\n{ctx}\n\n問題：{q}",
                  1024, 0.2, "你是嚴謹的文件問答助手，只依據提供的上下文作答。", True)
        return ans or "（模型未回傳內容）", src, f"{llm.model} · 檢索 {t1 - t0:.1f}s · 生成 {time.time() - t1:.1f}s"


# ── 匯出：Word / CSV（回傳 bytes，供 st.download_button 使用）──
def export_csv_bytes(records) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(records[0]))
    w.writeheader()
    w.writerows(records)
    return buf.getvalue().encode("utf-8-sig")  # utf-8-sig：Excel 開啟中文不亂碼


def _md_runs(par, text):  # 將 **粗體** 轉成 Word 粗體
    for i, part in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if part:
            par.add_run(part).bold = bool(i % 2)


def build_docx(records, with_src=True) -> Document:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Microsoft JhengHei", Pt(11)
    normal.element.rPr.rFonts.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia", "Microsoft JhengHei")
    doc.add_heading("RAG 文件問答紀錄", 0)
    meta = doc.add_paragraph(f"文件：{records[-1]['文件']}　·　匯出時間：{datetime.now():%Y-%m-%d %H:%M}　·　共 {len(records)} 題")
    meta.runs[0].font.color.rgb = RGBColor(0x6B, 0x72, 0x80)
    for n, r in enumerate(records, 1):
        doc.add_heading(f"Q{n}．{r['問題']}", level=1)
        info = doc.add_paragraph(f"{r['時間']}　·　{r['策略']}　·　{r['模型']}")
        info.runs[0].font.size, info.runs[0].font.color.rgb = Pt(9), RGBColor(0x6B, 0x72, 0x80)
        for line in r["回答"].splitlines():
            line = line.rstrip()
            if not line.strip():
                continue
            if m := re.match(r"^\s*#{1,6}\s+(.*)", line):
                doc.add_heading(m.group(1).replace("**", ""), level=2)
                continue
            if m := re.match(r"^(\s*)[-*•]\s+(.*)", line):
                par = doc.add_paragraph(style="List Bullet 2" if len(m.group(1)) >= 2 else "List Bullet")
                _md_runs(par, m.group(2))
            elif m := re.match(r"^\s*\d+[.)]\s+(.*)", line):
                par = doc.add_paragraph(style="List Number")
                _md_runs(par, m.group(1))
            else:
                _md_runs(doc.add_paragraph(), line)
        if with_src and r["參考片段"]:
            h = doc.add_paragraph()
            run = h.add_run("參考片段")
            run.bold, run.font.size = True, Pt(9)
            for blk in r["參考片段"].split("\n\n"):
                p = doc.add_paragraph(blk.strip())
                p.paragraph_format.left_indent = Pt(12)
                for run in p.runs:
                    run.font.size, run.font.color.rgb = Pt(8.5), RGBColor(0x55, 0x5B, 0x66)
    return doc


def build_docx_bytes(records, with_src=True) -> bytes:
    buf = io.BytesIO()
    build_docx(records, with_src).save(buf)
    return buf.getvalue()


# ── LINE Bot 投放 ──
def line_text(q, ans, limit=LINE_LIMIT):
    """組成 LINE 訊息：問題 + 回答（去除 Markdown 與 [片段n] 標註），總長不超過 limit 字。"""
    body = re.sub(r"[ \t]*\[片段[^\]]*\]", "", plain(ans))
    body = re.sub(r"[ \t]+(?=\n|$)", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    q = " ".join(q.split())
    msg = f"❓ {q[:59] + '…' if len(q) > 60 else q}\n\n{body}"
    if len(msg) > limit:
        msg = msg[:limit - 1].rstrip() + "…"   # 直接截到 limit 字（含「…」），不因句尾而縮短
    return msg


def plain(md: str) -> str:
    md = re.sub(r"\*\*(.+?)\*\*", r"\1", md)
    return re.sub(r"^\s*#{1,6}\s+", "", md, flags=re.M)


def line_send(token, text, host=LINE_HOST):
    """Broadcast 給所有加入此官方帳號的好友（會消耗每月訊息額度）。"""
    with ApiClient(Configuration(access_token=token, host=host)) as c:
        MessagingApi(c).broadcast(BroadcastRequest(messages=[TextMessage(text=text)]))


def line_check(token, host=LINE_HOST):
    with ApiClient(Configuration(access_token=token, host=host)) as c:
        return MessagingApi(c).get_bot_info().display_name


def line_err(e):
    if isinstance(e, ApiException):
        hint = {400: "請求格式錯誤", 401: "Token 無效或已過期", 403: "權限不足", 429: "超過訊息額度或請求過於頻繁"}.get(e.status, "")
        return f"HTTP {e.status} {hint}".strip()
    return f"{type(e).__name__}: {e}"


def push_line(token, q, ans) -> str:
    text = line_text(q, ans)
    try:
        line_send(token, text)
        return f"✓ 已投放到 LINE（{len(text)} 字）"
    except Exception as e:  # noqa: BLE001
        return f"✗ LINE 投放失敗：{line_err(e)}"


# ═══════════════ 介面 ═══════════════
st.set_page_config(page_title="RAG 文件問答", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="正在載入嵌入模型（首次需下載，約 470 MB）…")
def get_rag() -> RAG:
    rag = RAG()
    rag.embedder()
    return rag


rag = get_rag()
ss = st.session_state
for _k, _v in {"doc": None, "doc_tag": None, "doc_msg": "", "doc_error": "", "messages": [],
               "records": [], "last_qa": None, "model_options": MODELS, "is_admin": False,
               "pending": None}.items():
    ss.setdefault(_k, _v)
if "key_vault" not in ss:
    ss.key_vault = KeyVault()
ss.setdefault("key_error", "")
if "line_vault" not in ss:
    ss.line_vault = KeyVault()
ss.setdefault("line_error", "")


def _store_line():
    """LINE Token 送出時：加密保存並清空輸入框。"""
    raw = (ss.get("line_input") or "").strip()
    if raw:
        if len(raw) >= 40:
            ss.line_vault.put(raw)
            ss.line_error = ""
        else:
            ss.line_error = "Token 長度不符，請確認是否完整複製"
    ss.line_input = ""


def _store_key():
    """輸入框送出時：加密保存並立即清空輸入框，明文不留在 session 狀態中。"""
    raw = (ss.get("key_input") or "").strip()
    if raw:
        if raw.startswith("gsk_") and len(raw) >= 20:
            ss.key_vault.put(raw)
            ss.key_error = ""
        else:
            ss.key_error = "金鑰格式不符（Groq 金鑰應以 gsk_ 開頭）"
    ss.key_input = ""


# ── 側欄 ──
with st.sidebar:
    st.title("📚 RAG 文件問答")
    st.caption("Groq × ChromaDB · 線上版")

    # API Key：只存在使用者自己的 session，不寫入伺服器
    st.subheader("🔑 Groq API Key")
    vault: KeyVault = ss.key_vault
    if vault.has_key:
        st.success("✓ 金鑰已以 AES-256-GCM 加密保存於此工作階段")
        if st.button("清除金鑰"):
            vault.clear()
            st.rerun()
    else:
        st.caption("未輸入則使用伺服器預設金鑰")
    st.text_input("輸入 API Key（Enter 送出後加密並清空）", type="password", key="key_input",
                  placeholder="gsk_…", on_change=_store_key,
                  help="金鑰只保存在你的工作階段記憶體中，不寫入伺服器檔案或日誌。")
    if ss.key_error:
        st.error(ss.key_error)
    key = vault.get() or ENV_KEY
    if st.button("驗證並取得模型列表"):
        if not key:
            st.warning("尚未設定 Key")
        else:
            ms = list_models(key)
            if ms:
                ss.model_options = ms
                st.success(f"✓ Key 有效 · {len(ms)} 個模型")
            else:
                st.error("✗ Key 無效或無法取得模型")
    model = st.selectbox("模型", ss.model_options, key="model")

    # 文件
    st.subheader("📂 文件")
    up = st.file_uploader("選擇 PDF / DOCX", type=["pdf", "docx"], key="uploader")
    if up is not None:
        tag = f"{up.name}-{up.size}"
        if tag != ss.doc_tag:
            ss.doc_tag, ss.doc_error = tag, ""
            if up.size > MAX_UPLOAD_MB * 1024 * 1024:
                ss.doc_error = f"檔案超過 {MAX_UPLOAD_MB} MB 上限"
            else:
                with st.spinner("解析並建立索引中…"):
                    try:
                        new_doc, msg = rag.load(up.name, up.getvalue())
                    except ValueError as e:
                        ss.doc_error = str(e)
                    except Exception as e:  # noqa: BLE001
                        ss.doc_error = f"載入失敗：{type(e).__name__}: {e}"
                    else:
                        if ss.doc is not None:
                            rag.release(ss.doc.sid)   # 新文件成功後才釋放舊文件
                        ss.doc, ss.doc_msg = new_doc, msg
    if ss.doc_error:
        st.error(ss.doc_error)
    if ss.doc is None:
        st.caption("尚未載入文件")
    elif rag.is_alive(ss.doc.sid):
        st.success(ss.doc_msg)
    else:
        st.warning("文件索引已逾時釋放，請重新上傳")
    ready = ss.doc is not None and rag.is_alive(ss.doc.sid)

    # 檢索設定
    st.subheader("⚙️ 檢索設定")
    strategy = st.selectbox("檢索策略", list(STRATEGIES), format_func=lambda s: STRATEGIES[s][0], key="strategy")
    st.caption(STRATEGIES[strategy][1])
    k = st.slider("片段數量 Top-K", 1, 10, 4)

    # LINE Bot：Token 一律可輸入（加密保存）；群發投放需管理員密碼
    st.subheader("💬 LINE Bot")
    line_vault: KeyVault = ss.line_vault
    line_tok, auto_line = line_vault.get() or ENV_LINE, False
    if line_vault.has_key:
        st.success("✓ LINE Token 已加密保存於此工作階段")
        if st.button("清除 LINE Token"):
            line_vault.clear()
            st.rerun()
    elif ENV_LINE:
        st.caption("已使用伺服器設定的 LINE Token")
    else:
        st.caption("尚未設定 LINE Token")
    st.text_input("輸入 Channel Access Token（Enter 送出後加密並清空）", type="password", key="line_input",
                  placeholder="Channel Access Token", on_change=_store_line,
                  help="Token 只保存在你的工作階段記憶體中，不寫入伺服器檔案或日誌。")
    if ss.line_error:
        st.error(ss.line_error)
    if st.button("驗證 LINE Token"):
        if not line_tok:
            st.warning("尚未設定 LINE Token")
        else:
            try:
                st.success(f"✓ Token 有效 · {line_check(line_tok)}")
            except Exception as e:  # noqa: BLE001
                st.error(f"✗ Token 驗證失敗：{line_err(e)}")

    if not ADMIN_PASSWORD:
        st.warning("未設定 ADMIN_PASSWORD，群發投放已停用（請在 Secrets 中設定）")
    else:
        with st.expander("🔐 管理員投放控制"):
            pw = st.text_input("管理員密碼", type="password", key="admin_pw")
            ss.is_admin = bool(pw) and hmac.compare_digest(pw.encode(), ADMIN_PASSWORD.encode())
            if pw and not ss.is_admin:
                st.error("密碼錯誤")
            if ss.is_admin:
                auto_line = st.checkbox(f"回答後自動投放（≤{LINE_LIMIT} 字）")
                st.caption("⚠ 投放會群發給所有加入官方帳號的好友，並消耗每月訊息額度。")
                if ss.last_qa:
                    preview = line_text(*ss.last_qa)
                    st.caption(f"投放預覽（{len(preview)}/{LINE_LIMIT} 字）")
                    st.text(preview)
                else:
                    st.caption("尚無可投放的回答")
                confirm = st.checkbox("我確認要群發")
                if st.button("📤 投放最後一則回答", disabled=not (confirm and ss.last_qa and line_tok)):
                    st.toast(push_line(line_tok, *ss.last_qa))

    # 匯出
    st.subheader("💾 匯出紀錄")
    with_src = st.checkbox("Word 包含參考片段", value=True)
    if ss.records:
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        c1, c2 = st.columns(2)
        c1.download_button("📄 Word", build_docx_bytes(ss.records, with_src), f"RAG問答紀錄_{stamp}.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        c2.download_button("📊 CSV", export_csv_bytes(ss.records), f"RAG問答紀錄_{stamp}.csv", mime="text/csv")
    else:
        st.caption("尚無問答紀錄")
    if st.button("🗑 清除對話"):
        ss.messages, ss.records, ss.last_qa = [], [], None
        st.rerun()

# ── 主區 ──
st.header("📚 文件問答")
if not ready:
    st.info("請先在側欄上傳 PDF 或 DOCX 文件。")

if ready and not ss.messages:
    st.caption("試試看：")
    for i, (col, text) in enumerate(zip(st.columns(len(EXAMPLES)), EXAMPLES)):
        if col.button(text, key=f"ex{i}", use_container_width=True):
            ss.pending = text
            st.rerun()

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("timing"):
            st.caption(m["timing"])
        if m.get("src"):
            with st.expander("🔎 參考片段"):
                st.text(m["src"])
        if m.get("line"):
            st.caption(m["line"])

q = st.chat_input("輸入問題，Enter 送出", disabled=not ready)
q = q or ss.pending
ss.pending = None

if q and ready:
    ss.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            try:
                ans, src, timing = rag.answer(ss.doc, q, strategy, k, model, key)
            except Exception as e:  # noqa: BLE001
                ans, src, timing = f"✗ {type(e).__name__}: {e}", "", ""
        st.markdown(ans)
        if timing:
            st.caption(timing)
        if src:
            with st.expander("🔎 參考片段"):
                st.text(src)

    entry = {"role": "assistant", "content": ans, "timing": timing, "src": src, "line": ""}
    ss.records.append({"時間": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "文件": ss.doc.name, "問題": q,
                       "回答": ans, "策略": STRATEGIES[strategy][0], "模型": model or MODELS[0],
                       "參考片段": src.split("─" * 40, 1)[-1].strip()})
    if timing:  # timing 非空 = 模型確實產生了回答
        ss.last_qa = (q, ans)
        if ss.is_admin and auto_line and line_tok:
            entry["line"] = push_line(line_tok, q, ans)
            st.toast(entry["line"])
    ss.messages.append(entry)

st.caption("對話與文件索引只保存在伺服器記憶體中，程式重啟或閒置逾時後即清除。")
