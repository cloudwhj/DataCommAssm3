import os
import re
import io
import json
import time
import math
import glob
import faiss
import numpy as np
import streamlit as st
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import boto3
import re

# ---------------------------
# Configuration
# ---------------------------
APP_TITLE = "RMIT Academic Policies Assistant"
APP_DESC = "Get concise, clause-backed answers about academic policies (extensions, integrity, appeals, attendance, and more)."

DEFAULT_TOP_K = 6
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_MIN_SCORE = 0.35  # cosine threshold for safe answers

INDEX_DIR = "./data/index"
DATA_DIR = "./data/policies"
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
PASSAGES_PATH = os.path.join(INDEX_DIR, "passages.json")
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")

BEDROCK_REGION = os.getenv("BEDROCK_REGION", "ap-southeast-2")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

COGNITO_REGION = "ap-southeast-2"
BEDROCK_REGION = "ap-southeast-2"
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

IDENTITY_POOL_ID = "ap-southeast-2:eaa059af-fd47-4692-941d-e314f2bd5a0e"   # from your AWS setup
USER_POOL_ID = "ap-southeast-2_NfoZbDvjD"
APP_CLIENT_ID = "3p3lrenj17et3qfrnvu332dvka"

USERNAME = "cludwhj2003@gmail.com" # Replace with your username
PASSWORD = "dis5lik-E"    # Replace with your password

BLOCKLIST_PATTERNS = [
    r"^\s*User:\s*.*$",                      # lines starting with 'User:'
    r"^\s*Assistant:\s*.*$",                 # lines starting with 'Assistant:'
    r"^\s*Relevant policy excerpts:?[\s\S]*$",  # any dumped excerpt header + following content
]

def get_credentials(username, password):
    idp_client = boto3.client("cognito-idp", region_name=COGNITO_REGION)
    response = idp_client.initiate_auth(
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={"USERNAME": username, "PASSWORD": password},
        ClientId=APP_CLIENT_ID,
    )
    id_token = response["AuthenticationResult"]["IdToken"]

    identity_client = boto3.client("cognito-identity", region_name=COGNITO_REGION)
    identity_response = identity_client.get_id(
        IdentityPoolId=IDENTITY_POOL_ID,
        Logins={f"cognito-idp.{COGNITO_REGION}.amazonaws.com/{USER_POOL_ID}": id_token},
    )

    creds_response = identity_client.get_credentials_for_identity(
        IdentityId=identity_response["IdentityId"],
        Logins={f"cognito-idp.{COGNITO_REGION}.amazonaws.com/{USER_POOL_ID}": id_token},
    )
    return creds_response["Credentials"]


# ---------------------------
# Utilities: cleaning / schema
# ---------------------------
WS_RE = re.compile(r"\s+")

def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u00A0", " ")
    t = WS_RE.sub(" ", t).strip()
    return t

def normalize_clause_id(cid: str) -> str:
    cid = str(cid or "").strip()
    return cid

def clean_clause(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": normalize_clause_id(c.get("id", "")),
        "text": clean_text(c.get("text", "")),
        "tags": sorted(list(set([clean_text(x).lower() for x in (c.get("tags") or []) if x])))
    }

def load_policy_json_from_bytes(
    name: str,
    b: bytes,
    *,
    promote_subclauses: bool = True,
    min_clause_len: int = 20
) -> Dict[str, Any]:
    """
    Loader for your policy JSON schema:

    {
      "metadata": {
        "title": "...",
        "review_date": "YYYY-MM-DD",   # or "approval_date"
        "source_path": "https://...",
        ...
      },
      "structure": [
        {
          "part_title": "...",
          "sections": [
            {
              "section_title": "...",
              "clauses": [
                {
                  "clause_number": "10",
                  "text": "clause text ...",
                  "subclauses": ["text a", "text b", ...]   # optional
                },
                ...
              ]
            },
            ...
          ]
        },
        ...
      ],
      "qa_index": [...]  # ignored here
    }
    """
    try:
        raw = json.loads(b.decode("utf-8"))
    except Exception:
        return {}

    meta = raw.get("metadata", {}) or {}
    source = clean_text(meta.get("title", name))
    # prefer review_date, fallback to approval_date, else empty
    version = clean_text(meta.get("review_date") or meta.get("approval_date") or "")
    url = clean_text(meta.get("source_path", ""))

    parts = raw.get("structure", []) or []
    cleaned_sections: List[Dict[str, Any]] = []

    for part in parts:
        sections = part.get("sections", []) or []
        for sec in sections:
            heading = clean_text(sec.get("section_title", ""))
            out_clauses: List[Dict[str, Any]] = []

            for cl in sec.get("clauses", []) or []:
                base_id = normalize_clause_id(cl.get("clause_number"))
                base_text = clean_text(cl.get("text", ""))

                if len(base_text) >= min_clause_len:
                    out_clauses.append({
                        "id": str(base_id),
                        "text": base_text,
                        "tags": []
                    })

                # Optionally promote subclauses (10(a), 10(b), ...)
                if promote_subclauses:
                    subs = cl.get("subclauses") or []
                    for i, sub in enumerate(subs):
                        stxt = clean_text(sub)
                        if len(stxt) >= min_clause_len:
                            sid = f"{base_id}({chr(97 + i)})"  # a, b, c...
                            out_clauses.append({
                                "id": str(sid),
                                "text": stxt,
                                "tags": []
                            })

            # Deduplicate within the section by (id, text)
            if out_clauses:
                seen = set()
                dedup = []
                for c in out_clauses:
                    key = (c["id"], c["text"])
                    if key not in seen:
                        seen.add(key)
                        dedup.append(c)
                if dedup:
                    cleaned_sections.append({"heading": heading, "clauses": dedup})

    if not cleaned_sections:
        return {}

    return {
        "source": source,
        "version": version,
        "url": url,
        "sections": cleaned_sections
    }


def flatten_passages(policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert policy JSON into flat clause-level passages for embedding,
    auto-generating keyword tags from section/part titles and clause text.
    """
    keyword_map = {
        "extension": ["extension", "late", "late submission", "7 days", "elp"],
        "special_consideration": ["special consideration", "consideration", "longer than 7", "ongoing"],
        "integrity": ["plagiarism", "collusion", "integrity", "cheating"],
        "submission": ["submit", "submission", "canvas", "turnitin", "upload"],
        "feedback": ["feedback", "rubric", "criteria", "comments"],
        "appeals": ["appeal", "grade appeal", "final grade", "review result", "complaint", "college appeals committee"],
        "attendance": ["attendance", "absent", "tutorial", "lecture", "lab"],
        "conduct": ["conduct", "complaint", "behavior", "behaviour"]
    }

    passages = []
    for p in policies:
        for s in p["sections"]:
            sec_name = s.get("heading", "").lower()
            for c in s["clauses"]:
                txt = c["text"].lower()
                inferred_tags = set()

                # Infer from section title
                for kw_group, words in keyword_map.items():
                    if any(w in sec_name for w in words) or any(w in txt for w in words):
                        inferred_tags.add(kw_group)

                # Merge with any explicit tags
                explicit_tags = [t.lower() for t in c.get("tags", []) if t]
                all_tags = sorted(list(inferred_tags.union(explicit_tags)))

                passages.append({
                    "text": c["text"],
                    "meta": {
                        "source": p["source"],
                        "version": p["version"],
                        "url": p["url"],
                        "heading": s["heading"],
                        "clause_id": c["id"],
                        "tags": all_tags
                    }
                })
    return passages


# ---------------------------
# Embeddings + Index
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_embedder(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

def build_faiss_index(texts: List[str], embedder) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    if not texts:
        return None, None
    embs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    return index, embs

def save_index(index: faiss.IndexFlatIP, passages: List[Dict[str, Any]]):
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(PASSAGES_PATH, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)
    manifest = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "num_passages": len(passages),
        "policies": sorted(list({p["meta"]["source"] for p in passages})),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def load_saved_index():
    if not (os.path.exists(FAISS_INDEX_PATH) and os.path.exists(PASSAGES_PATH) and os.path.exists(MANIFEST_PATH)):
        return None, None, None
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
        passages = json.load(f)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return index, passages, manifest

# ---------------------------
# Retrieval (top-k + threshold)
# ---------------------------
def search(index: faiss.IndexFlatIP, embedder, passages: List[Dict[str, Any]], query: str, k: int) -> List[Dict[str, Any]]:
    if not index or not passages:
        return []
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    k = max(1, min(k, len(passages)))
    D, I = index.search(q, k)
    hits = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        p = passages[idx]
        hits.append({"text": p["text"], "score": float(score), "meta": p["meta"]})
    return hits

def format_citation(m: Dict[str, Any]) -> str:
    src = m.get("source", "Policy")
    cid = m.get("clause_id", "?")
    heading = m.get("heading", "")
    if heading:
        return f"{src}, Clause {cid} ({heading})"
    return f"{src}, Clause {cid}"

# ---------------------------
# NLP-lite: topic/entity extraction, query rewrite, RRF fusion
# ---------------------------

TOPIC_KEYWORDS = {
    "extensions": ["extension", "extensions", "late", "late submission", "7 days", "seven days"],
    "special_consideration": ["special consideration", "special-consideration", "longer than 7", "longer than seven", "serious", "ongoing"],
    "integrity": ["plagiarism", "collusion", "academic integrity", "cheating", "Turnitin"],
    "appeals": ["appeal", "grade appeal", "final grade", "appeals", "review"],
    "attendance": ["attendance", "absent", "absence", "tutorial", "lab", "lecture"],
    "conduct": ["conduct", "complaint", "complaints", "behavior", "behaviour"],
}

ASSESSMENT_TYPES = ["assignment", "exam", "quiz", "test", "presentation", "project", "lab", "tutorial"]
EVIDENCE_TERMS   = ["medical", "certificate", "documentation", "evidence", "elp", "els"]

def classify_topic(text: str) -> str | None:
    t = text.lower()
    best, max_hits = None, 0
    for topic, kws in TOPIC_KEYWORDS.items():
        hits = sum(1 for k in kws if k in t)
        if hits > max_hits:
            best, max_hits = topic, hits
    return best

def extract_entities(text: str) -> dict:
    t = text.lower()
    ents = {
        "days": None,
        "assessment_type": None,
        "evidence": any(k in t for k in EVIDENCE_TERMS),
    }
    # days (simple patterns like "7 days", "10-day")
    import re
    m = re.search(r"(\d{1,2})\s*day", t)
    if m:
        ents["days"] = int(m.group(1))
    # assessment type
    for a in ASSESSMENT_TYPES:
        if a in t:
            ents["assessment_type"] = a
            break
    return ents

def update_dialog_state(user_text: str):
    topic = classify_topic(user_text) or st.session_state.get("topic")
    ents = extract_entities(user_text)
    # merge entities
    prev = st.session_state.get("entities") or {}
    merged = {**prev, **{k:v for k,v in ents.items() if v not in [None, False]}}
    st.session_state.topic = topic
    st.session_state.entities = merged

def rewrite_query(user_text: str) -> str:
    """
    Turn vague follow-ups like 'can u explain in detail'
    into a fully qualified question using remembered topic/entities.
    """
    t = user_text.strip()
    if not t:
        return t

    topic = st.session_state.get("topic")
    ents = st.session_state.get("entities") or {}

    # If the user message is obviously vague, build a new clarified query
    vague_phrases = [
        "explain in detail",
        "explain more",
        "can you explain more",
        "what do you mean",
        "how does that work",
        "can u explain in detail",
        "tell me more",
        "more detail",
        "go deeper"
    ]

    lower_t = t.lower()
    is_vague = any(phrase in lower_t for phrase in vague_phrases)

    if is_vague and topic:
        detail_bits = []

        # map topic to human-readable label
        topic_label_map = {
            "appeals": "appealing a final course result through review and the College Appeals Committee process",
            "extensions": "requesting an assessment deadline extension",
            "special_consideration": "special consideration for serious circumstances",
            "integrity": "academic integrity and plagiarism rules",
            "attendance": "attendance/participation requirements",
            "conduct": "student conduct or harassment reporting",
        }
        nice_topic = topic_label_map.get(topic, topic.replace("_", " "))

        # include known entities if present
        if ents.get("days"):
            detail_bits.append(f"{ents['days']}-day extension request")
        if ents.get("assessment_type"):
            detail_bits.append(f"for a {ents['assessment_type']} assessment")

        tail = ""
        if detail_bits:
            tail = " regarding " + ", ".join(detail_bits)

        # build clarified query
        return (
            f"In the context of {nice_topic}{tail}, please explain the full process in more detail, "
            f"including steps, requirements, timelines, and who makes the decision."
        )

    # default: keep original but add topic hint if available
    if topic and topic not in lower_t:
        return f"{t} (This is about {topic}.)"

    return t


def rrf_merge(hit_lists: list[list[dict]], k: int = 8, k_rank: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion for multiple retrievals.
    hit_lists = [hits_from_query1, hits_from_query2, ...]
    Each hit is {'text','score','meta'}
    """
    from collections import defaultdict
    scores = defaultdict(float)
    seen_meta = {}  # keep one exemplar per unique (source, clause_id, text)
    def key(h):
        m = h["meta"]; return (m.get("source"), m.get("clause_id"), h["text"])
    for hits in hit_lists:
        for r, h in enumerate(hits):
            scores[key(h)] += 1.0 / (k_rank + r + 1)
            seen_meta[key(h)] = h
    fused = [{**seen_meta[k_], "rrf": v} for k_, v in scores.items()]
    fused.sort(key=lambda x: x["rrf"], reverse=True)
    return fused[:k]


# ---------------------------
# Conversation memory
# ---------------------------
def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of {role: "user"/"assistant", "content": str, "citations": [..]}
    if "summary" not in st.session_state:
        st.session_state.summary = ""
    if "index" not in st.session_state:
        st.session_state.index = None
    if "passages" not in st.session_state:
        st.session_state.passages = None
    if "manifest" not in st.session_state:
        st.session_state.manifest = None
    if "topic" not in st.session_state:
        st.session_state.topic = None
    if "entities" not in st.session_state:
        st.session_state.entities = {}


def add_message(role: str, content: str, citations: List[str] = None):
    st.session_state.messages.append({"role": role, "content": content, "citations": citations or []})
    # compress summary every few turns (simple heuristic)
    if len(st.session_state.messages) % 6 == 0:
        st.session_state.summary = compress_summary(st.session_state.summary, st.session_state.messages[-6:])

def compress_summary(previous_summary: str, recent_msgs: List[Dict[str, Any]]) -> str:
    # Lightweight heuristic compression without extra model calls
    # Keep only salient user intents and assistant conclusions (first sentence)
    lines = []
    if previous_summary:
        lines.append(previous_summary.strip())
    for m in recent_msgs:
        txt = m["content"].strip()
        if m["role"] == "user":
            lines.append(f"User: {txt[:180]}")
        else:
            first = txt.split(".")[0] if "." in txt else txt[:180]
            lines.append(f"Assistant: {first}.")
    summary = " ".join(lines)
    return summary[-2000:]  # cap size

def conversation_context(n_recent: int = 4) -> str:
    # Build a compact string with summary + last n turns
    ctx = []
    if st.session_state.summary:
        ctx.append(f"Conversation summary: {st.session_state.summary}")
    for m in st.session_state.messages[-n_recent:]:
        role = m["role"].capitalize()
        ctx.append(f"{role}: {m['content']}")
    return "\n".join(ctx)

# ---------------------------
# Prompting (guardrails + dynamic)
# ---------------------------
SYSTEM_PROMPT = """
You are RMIT's Academic Policy Assistant. Your scope is limited to official academic policies (e.g., Assessment & Assessment Flexibility, Academic Integrity, Appeals & Complaints, Course Withdrawal & Academic Progress, Attendance & Participation).

Core rules:
1) Grounding & citations: Quote or paraphrase only content supported by the indexed policy corpus. After each factual statement or bullet, include bracketed citations like [Policy Name, Clause X]. If multiple clauses apply, cite each.
2) No hallucinations: If you cannot find relevant clauses, say: "I can’t find this in the policies I have." and suggest contacting Student Connect.
3) Safety: Do NOT provide medical, legal, immigration, or personal advice. Do NOT impersonate staff or invent processes. For sensitive topics, redirect to official channels.
4) Privacy: Do not store or repeat personal identifiers. Mask any emails/IDs if user includes them.
5) Neutral, concise, helpful tone. Ask one clarifying question when the query is ambiguous or missing critical details.
6) Conflicts: If policies seem conflicting, present both sides with citations and suggest contacting Student Connect for confirmation.
7) Formatting: Prefer short paragraphs or bullet points. Always include citations at the end of lines.
8) Do NOT repeat or quote the raw context text.
9) Do NOT print the words "User:", "Assistant:", "Relevant policy excerpts", or any XML tags I give you.

Output requirements:
- Direct, concise answer only (no preambles, no bullet of raw excerpts).
- Include citations inline for factual statements.
"""

def build_user_prompt(user_q: str, retrieved: List[Dict[str, Any]], min_score: float, conv_ctx: str) -> Tuple[str, List[str]]:
    """
    Builds the full text prompt for Bedrock.
    Keeps retrieved context hidden from the user (used only by the model),
    while citations are extracted separately for display.
    """
    kept = [h for h in retrieved if h["score"] >= min_score]
    citations = []
    context_lines = []
    for h in kept:
        m = h["meta"]
        cite = format_citation(m)
        citations.append(cite)
        context_lines.append(f"[{cite}] {h['text']}")

    # hidden context for model
    hidden_context = "\n".join(context_lines) if context_lines else "None found above threshold."

    # the model sees everything, user only sees answer
    prompt = (
        f"{SYSTEM_PROMPT.strip()}\n\n"
        f"{conv_ctx}\n\n"
        f"<policy_context>\n{hidden_context}\n</policy_context>\n\n"
        f"Question: {user_q}\n\n"
        "Use ONLY the information inside <policy_context> plus the chat history to answer. "
        "Never print or list the contents of <policy_context>. "
        "Answer directly in 1–6 sentences with bracketed citations."
    )
    return prompt, citations

def sanitize_model_answer(text: str) -> str:
    # Remove XML tags if model prints them anyway
    text = re.sub(r"</?policy_context>", "", text, flags=re.IGNORECASE)
    # Drop any blocklisted lines/sections
    for pat in BLOCKLIST_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE)
    # Trim extra whitespace
    return re.sub(r"\n{3,}", "\n\n", text).strip()

# ---------------------------
# Bedrock (Anthropic Messages API)
# ---------------------------
def bedrock_client():
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

def invoke_bedrock(system_prompt, user_content, model_id, temperature, top_p, max_tokens=700):
    credentials = get_credentials(USERNAME, PASSWORD)
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=BEDROCK_REGION,
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretKey"],
        aws_session_token=credentials["SessionToken"],
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": system_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "messages": [{"role": "user", "content": user_content}],
    }
    response = bedrock_runtime.invoke_model(
        body=json.dumps(body),
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


# ---------------------------
# UI Helpers
# ---------------------------
def freshness_badge(manifest: Dict[str, Any]):
    if not manifest:
        st.markdown(":red[Not indexed]")
        return
    ts = manifest.get("updated_at")
    num = manifest.get("num_passages", 0)
    policies = manifest.get("policies", [])
    st.markdown(
        f"**Data freshness:** {ts}  •  **Passages:** {num}  •  **Policies:** {', '.join(policies) if policies else '—'}"
    )

def show_sources(hits: List[Dict[str, Any]]):
    with st.expander("Sources & Retrieved Clauses"):
        if not hits:
            st.write("No clauses retrieved.")
            return
        for h in hits:
            m = h["meta"]
            st.markdown(f"- **{format_citation(m)}**  \nScore: `{h['score']:.3f}`  \nText: {h['text']}")

def follow_up_suggestions(user_q: str) -> List[str]:
    q = user_q.lower()
    sug = []
    if "extension" in q:
        sug = ["What if I need more than 7 days?", "What evidence is acceptable?", "How long is the decision time?"]
    elif "integrity" in q or "plagiarism" in q:
        sug = ["What counts as collusion?", "How is plagiarism investigated?", "What are potential outcomes?"]
    elif "appeal" in q:
        sug = ["What is the appeal window?", "Who reviews appeals?", "What evidence do I need?"]
    elif "attendance" in q:
        sug = ["Is attendance mandatory for labs?", "How are absences handled?", "Do I need medical certificates?"]
    else:
        sug = ["How to contact Student Connect?", "Where to find program coordinators?", "Show me a policy example."]
    return sug

# ---------------------------
# App
# ---------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session()

    # ---------- Header ----------
    st.title(APP_TITLE)
    st.caption(APP_DESC)
    st.info(
        "Disclaimer: This assistant quotes/paraphrases official RMIT policies and provides clause-level citations. "
        "If the answer cannot be located in the policies indexed, it will say so and suggest contacting Student Connect."
    )

    # ---------- Sidebar ----------
    with st.sidebar:
        st.subheader("Ask effectively")
        st.markdown("- Example: *How do I request an assignment extension?*")
        st.markdown("- Example: *What happens if I fail my assessment?*")
        st.markdown("- Example: *How do I appeal my final grade?*")

        st.divider()
        st.subheader("Knowledge Base")
        # 🔹 Ensure index is loaded before showing status
        if st.session_state.index is None or st.session_state.passages is None:
            idx, passages, manifest = load_saved_index()
            st.session_state.index = idx
            st.session_state.passages = passages
            st.session_state.manifest = manifest

        # 🔹 Now show correct freshness badge
        if st.session_state.index:
            freshness_badge(st.session_state.manifest)
        else:
            st.warning("⚠️ Knowledge base not indexed yet.")
        uploaded = st.file_uploader("Upload Academic Policy JSON files", type=["json"], accept_multiple_files=True)
        if st.button("Rebuild Index", use_container_width=True):
            if not uploaded:
                st.warning("Please upload at least one JSON policy file.")
            else:
                # Load + clean
                policies = []
                for f in uploaded:
                    data = load_policy_json_from_bytes(f.name, f.read())
                    if data:
                        policies.append(data)
                if not policies:
                    st.error("No valid policy data found in uploaded files.")
                else:
                    passages = flatten_passages(policies)
                    if not passages:
                        st.error("No passages created from uploaded policies.")
                    else:
                        embedder = get_embedder()
                        texts = [p["text"] for p in passages]
                        with st.spinner("Building embeddings & index..."):
                            index, _ = build_faiss_index(texts, embedder)
                        if not index:
                            st.error("Failed to build index.")
                        else:
                            save_index(index, passages)
                            st.session_state.index, st.session_state.passages, st.session_state.manifest = load_saved_index()
                            st.success("Index rebuilt successfully.")

        st.caption("Tip: Rebuilding the index updates the 'Data freshness' badge.")

    # ---------- Load index if available ----------
    if st.session_state.index is None:
        idx, passages, manifest = load_saved_index()
        st.session_state.index = idx
        st.session_state.passages = passages
        st.session_state.manifest = manifest

    top_k = DEFAULT_TOP_K
    min_score = DEFAULT_MIN_SCORE
    temperature = DEFAULT_TEMPERATURE
    top_p = DEFAULT_TOP_P
    model_id = BEDROCK_MODEL_ID

    # ---------- Layout columns ----------
    cols_top = st.columns([1, 4, 1])
    with cols_top[1]:
        if st.button("🗘 Start New Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.summary = ""
            st.success("Session cleared.")


    # Chat area
    st.subheader("Chat")

    # Taller chat window + sticky input
    st.markdown(
        """
        <style>
            .chat-window {
                height: 78vh;
                overflow-y: auto;
                padding: 0.75rem 1rem;
                border: 1px solid rgba(128,128,128,0.25);
                border-radius: 12px;
                background: rgba(240, 240, 240, 0.25);
            }
            .msg {
                margin: 0.5rem 0;
                padding: 0.75rem 0.9rem;
                border-radius: 10px;
                line-height: 1.35;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
            .msg-user {
                background: rgba(0, 120, 212, 0.10);
                border: 1px solid rgba(0, 120, 212, 0.20);
            }
            .msg-assistant {
                background: rgba(0, 0, 0, 0.03);
                border: 1px solid rgba(128,128,128,0.20);
            }
            .msg small {
                display: block;
                margin-top: 0.35rem;
                opacity: 0.65;
                font-size: 0.85em;
            }
            /* ensure space for sticky input */
            div.block-container { padding-bottom: 7rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

        # Render messages into the fixed-height window
    chat_html = ["<div class='chat-window'>"]
    for m in st.session_state.messages:
        role_class = "msg-user" if m["role"] == "user" else "msg-assistant"
        citations = ""
        if m["role"] == "assistant" and m.get("citations"):
            citations = f"<small>Citations: {' • '.join(m['citations'])}</small>"
        # Escape/format minimal; Streamlit markdown handles basic safety—this is static HTML block
        content_html = st._escape_markdown(m["content"]) if hasattr(st, "_escape_markdown") else m["content"]
        chat_html.append(f"<div class='msg {role_class}'>{content_html}{citations}</div>")
    chat_html.append("</div>")
    st.markdown("".join(chat_html), unsafe_allow_html=True)

    # Input at the bottom
    user_q = st.chat_input("Ask about academic policies...")
    if user_q:
        # Update dialog state (topic/entities)
        update_dialog_state(user_q)

        # Record the raw user message
        add_message("user", user_q)

        # Build a rewritten query using context
        q_rewrite = rewrite_query(user_q)

        # Retrieval (multi-signal): original, rewritten, and conversation summary
        hits = []; hits_rw = []; hits_sum = []
        t0 = time.time()
        if st.session_state.index and st.session_state.passages:
            embedder = get_embedder()
            hits     = search(st.session_state.index, embedder, st.session_state.passages, user_q, top_k)
            hits_rw  = search(st.session_state.index, embedder, st.session_state.passages, q_rewrite, top_k)
            # Use short conversation summary as an anchor query if available
            sum_q = st.session_state.summary[-280:] if st.session_state.summary else ""
            if sum_q:
                hits_sum = search(st.session_state.index, embedder, st.session_state.passages, sum_q, max(3, top_k//2))
        t_retrieval = (time.time() - t0) * 1000

        # Fuse lists via RRF
        fused = rrf_merge([hits, hits_rw, hits_sum], k=max(6, top_k))
        strong_hits = [h for h in fused if h["score"] >= min_score] if fused else []

        # Build prompt with fused hits
        ctx = conversation_context(n_recent=4)
        prompt, citations = build_user_prompt(q_rewrite, (strong_hits or fused), min_score, ctx)

        # Clarification fallback: if we have weak evidence AND the question is underspecified, ask a targeted question
        need_clarify = (not strong_hits) or (len(strong_hits) < 2)
        missing_slots = []
        # Topic-specific slots to clarify
        if (st.session_state.topic == "extensions" or "extension" in user_q.lower()):
            if st.session_state.entities.get("days") is None:
                missing_slots.append("how many days you need")
            if st.session_state.entities.get("assessment_type") is None:
                missing_slots.append("which assessment type (e.g., assignment, exam)")
        if st.session_state.topic == "attendance" and st.session_state.entities.get("assessment_type") is None:
            missing_slots.append("which class type (lecture, tutorial, lab)")

        if need_clarify and missing_slots:
            q = " and ".join(missing_slots)
            clarify = f"To give the most accurate clause, could you clarify {q}?"
            add_message("assistant", clarify, citations=[])
            st.rerun()

        # If still nothing strong, provide safe refusal with sources (if any)
        if not strong_hits:
            fallback_hits = fused[:3] if fused else []
            if fallback_hits:
                ctx = conversation_context(n_recent=4)
                prompt, citations = build_user_prompt(q_rewrite, fallback_hits, min_score, ctx)
                try:
                    model_answer = invoke_bedrock(
                        system_prompt=SYSTEM_PROMPT,
                        user_content=prompt,
                        model_id=model_id,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=700,
                )
                except Exception as e:
                    model_answer = f"Error contacting Bedrock: {e}"

                model_answer = sanitize_model_answer(model_answer)
                add_message("assistant", model_answer, citations=citations)
                st.rerun()
            else:
                # true no-context situation
                answer = (
                    "I need a bit more detail to pull the right policy. "
                    "Please clarify your question with more details. "
                    "or please consider contacting Student Connect: 1300 ASK RMIT or reach out to them via 'https://www.rmit.edu.au/students/support-services/student-connect'."
                )
                add_message("assistant", answer, citations=[])
                st.rerun()
        
        # Call Bedrock with guard-railed system prompt
        t1 = time.time()
        try:
            model_answer = invoke_bedrock(
                system_prompt=SYSTEM_PROMPT,
                user_content=prompt,
                model_id=model_id,
                temperature=temperature,
                top_p=top_p,
                max_tokens=700,
            )
        except Exception as e:
            model_answer = f"Error contacting Bedrock: {e}"
        t_gen = (time.time() - t1) * 1000

        model_answer = sanitize_model_answer(model_answer)
        add_message("assistant", model_answer, citations=citations)
        add_message("assistant", f"Retrieval: {t_retrieval:.0f} ms • Generation: {t_gen:.0f} ms", citations=[])
        st.rerun()


if __name__ == "__main__":
    main()
