# rag_policies.py
import json, os, re, glob
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

WHITESPACE_RE = re.compile(r"\s+")

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u00A0", " ")            # NBSP -> space
    t = WHITESPACE_RE.sub(" ", t).strip()   # collapse whitespace
    return t

def clean_clause(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(c.get("id", "")).strip(),
        "text": clean_text(c.get("text", "")),
        "tags": sorted(list(set([clean_text(x).lower() for x in (c.get("tags") or []) if x])))
    }

def load_policy_jsons(path_or_folder: str) -> List[Dict[str, Any]]:
    files = []
    if os.path.isdir(path_or_folder):
        files = glob.glob(os.path.join(path_or_folder, "*.json"))
    elif os.path.isfile(path_or_folder):
        files = [path_or_folder]
    else:
        return []

    policies = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            source = clean_text(data.get("source", os.path.basename(f)))
            version = clean_text(data.get("version", ""))
            url = clean_text(data.get("url", ""))
            sections = data.get("sections", [])
            cleaned_sections = []
            for s in sections:
                heading = clean_text(s.get("heading", ""))
                clauses = [clean_clause(c) for c in (s.get("clauses") or [])]
                # drop empty / tiny clauses
                clauses = [c for c in clauses if len(c["text"]) > 20]
                # dedupe identical texts within the section
                seen = set()
                dedup = []
                for c in clauses:
                    key = (c["id"], c["text"])
                    if key not in seen:
                        seen.add(key)
                        dedup.append(c)
                if dedup:
                    cleaned_sections.append({"heading": heading, "clauses": dedup})
            if cleaned_sections:
                policies.append({
                    "source": source, "version": version, "url": url, "sections": cleaned_sections
                })
        except Exception:
            # skip bad files silently for robustness in demo
            continue
    return policies

def flatten_passages(policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Turn policy JSON -> list of passages with citation info."""
    passages = []
    for p in policies:
        for s in p["sections"]:
            for c in s["clauses"]:
                passages.append({
                    "text": f"{c['text']}",
                    "meta": {
                        "source": p["source"],
                        "version": p["version"],
                        "url": p["url"],
                        "heading": s["heading"],
                        "clause_id": c["id"],
                        "tags": c["tags"]
                    }
                })
    return passages

class PolicyRAG:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.passages = []  # keep to cite later

    def build(self, passages: List[Dict[str, Any]]):
        self.passages = passages
        texts = [p["text"] for p in passages]
        if not texts:
            self.index = None
            return
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        d = embs.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embs)

    def search(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        if not self.index or not self.passages:
            return []
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q, min(k, len(self.passages)))
        hits = []
        for idx, score in zip(I[0], D[0]):
            if idx == -1: 
                continue
            p = self.passages[idx]
            hits.append({
                "text": p["text"],
                "score": float(score),
                "meta": p["meta"]
            })
        return hits

def render_rag_prompt(user_q: str, hits: List[Dict[str, Any]]) -> str:
    # Keep prompt small and cite properly
    context_lines = []
    for h in hits:
        m = h["meta"]
        cite = f"{m['source']} (Clause {m['clause_id']}, {m['heading']})"
        context_lines.append(f"- [{cite}] {h['text']}")
    context = "\n".join(context_lines)
    system = (
        "You are an assistant that answers questions ONLY using RMIT Academic Policies. "
        "Cite the policy and clause in brackets like [Policy Name, Clause X]. "
        "If the answer is not covered, say you cannot find it and suggest contacting Student Connect."
    )
    return (
        f"{system}\n\n"
        f"User question:\n{user_q}\n\n"
        f"Relevant policy excerpts:\n{context}\n\n"
        f"Answer concisely with citations."
    )
