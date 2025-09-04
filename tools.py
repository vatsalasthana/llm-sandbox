import os, re, glob, math, requests
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TOOL_REGISTRY = {}

def register_tool(name: str):
    def wrapper(fn):
        _TOOL_REGISTRY[name] = fn
        return fn
    return wrapper

def call_tool(name: str, input_str: str) -> str:
    if name not in _TOOL_REGISTRY:
        return f"Tool '{name}' not found."
    return _TOOL_REGISTRY[name](input_str)

# ---------------- Calculator ----------------
_ALLOWED = re.compile(r"^[0-9\s\(\)\+\-\*\/\.%]+$")
@register_tool("calculator")
def calculator(expr: str) -> str:
    if not _ALLOWED.match(expr):
        return "Calculator error: disallowed characters."
    try:
        result = eval(expr, {"__builtins__": None, "math": math}, {})
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"

# ---------------- Local RAG ----------------
class LocalRAG:
    def __init__(self, kb_dir: str = "knowledge_base", chunk_size=800, chunk_overlap=120):
        self.kb_dir = kb_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = []
        self._fit()

    def _chunk_text(self, text: str) -> List[str]:
        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i:i+self.chunk_size]
            chunks.append(chunk)
            i += max(1, self.chunk_size - self.chunk_overlap)
        return chunks

    def _fit(self):
        files = glob.glob(os.path.join(self.kb_dir, "**", "*.*"), recursive=True)
        texts = []
        self.docs = []
        for fp in files:
            if any(fp.lower().endswith(ext) for ext in [".txt", ".md"]):
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        raw = f.read()
                    chunks = self._chunk_text(raw)
                    for c in chunks:
                        self.docs.append((fp, c))
                        texts.append(c)
                except Exception:
                    pass
        if self.docs:
            self.vectorizer = TfidfVectorizer().fit(texts)
            self.matrix = self.vectorizer.transform(texts)
        else:
            self.vectorizer = None
            self.matrix = None

    def query(self, q: str, top_k=4) -> str:
        if not self.docs:
            return "RAG: knowledge base empty."
        q_vec = self.vectorizer.transform([q])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        idxs = sims.argsort()[::-1][:top_k]
        results = []
        for i in idxs:
            fp, chunk = self.docs[i]
            results.append(f"[{fp}]{chunk}")
        return "\n---\n".join(results)

RAG = LocalRAG()
@register_tool("rag")
def rag_tool(query: str) -> str:
    return RAG.query(query)

# ---------------- Live Search (Serper.dev) ----------------
@register_tool("live_search")
def live_search(query: str) -> str:
    """
    Perform a live web search using Serper.dev API.
    Requires SERPER_API_KEY in environment.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Live search error: SERPER_API_KEY not set."

    try:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": 5}

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("organic", [])[:5]:
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            results.append(f"{title}\n{snippet}\n{link}")
        
        return "\n\n---\n\n".join(results) if results else "No results found."

    except Exception as e:
        return f"Live search error: {e}"
