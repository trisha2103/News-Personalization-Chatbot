# -------------------- app_streamlit.py --------------------
# 1) FIRST Streamlit call must be set_page_config
import streamlit as st
st.set_page_config(page_title="News Chatbot", page_icon="🗞️", layout="centered")

# 2) Rest of imports
import json, sqlite3, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# 3) Load artifacts made in Steps 3–6
ART = Path("artifacts")
DB  = Path("data/news.db")
meta = pd.read_csv(ART/"news_meta.csv")
info = json.load(open(ART/"index_info.json"))
dim  = int(info["dim"])
embs = np.memmap(ART/"news_embs.dat", dtype="float32", mode="r", shape=(len(meta), dim))
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

# 4) Build user profiles (cached)
@st.cache_resource
def build_profiles():
    if not DB.exists(): return {}
    id2row = {nid:i for i, nid in enumerate(meta.news_id)}
    conn = sqlite3.connect(str(DB))
    df = pd.read_sql_query("""
      SELECT im.user_id, it.news_id, it.label
      FROM impression_items it
      JOIN impressions im ON im.impression_id = it.impression_id
      WHERE im.impression_id LIKE 'train_%'
    """, conn)
    conn.close()
    df = df[df.news_id.isin(id2row)]
    clicks = df[df.label==1].groupby("user_id")["news_id"].apply(list)
    prof = {}
    for u, nids in clicks.items():
        rows = [id2row[n] for n in nids if n in id2row]
        if not rows: 
            continue
        v = np.mean(embs[rows], axis=0)
        n = np.linalg.norm(v)
        if n > 0: 
            prof[u] = (v/n).astype("float32")
    return prof

profiles = build_profiles()
DEFAULT_USER = next(iter(profiles.keys()), None)

# 5) Retrieval + summarization
def _clean_vec(v):
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
    n = np.linalg.norm(v);  return v if n==0 else (v/n)

def retrieve_personalized(query: str, user_id: str|None, k: int = 5, alpha: float = 0.6):
    qv = _clean_vec(model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0])
    base = embs @ qv
    uv = profiles.get(user_id) if user_id else None
    scores = base if uv is None else (alpha*base + (1-alpha)*(embs @ _clean_vec(uv)))
    k = min(k, len(meta))
    idx = np.argpartition(-scores, k-1)[:k]; idx = idx[np.argsort(-scores[idx])]
    out = meta.iloc[idx].copy(); out["score"] = [float(scores[i]) for i in idx]
    return out[["score","title","abstract","url","category"]]

def summarize(query: str, hits: pd.DataFrame, max_sents=5):
    import numpy as np, re
    if hits.empty:
        return "I couldn't find matching articles.", []

    def to_text(x):
        # turn NaN/None/float into a clean string
        if x is None: return ""
        if isinstance(x, float):
            return "" if np.isnan(x) else str(x)
        return str(x)

    sents, src, titles = [], [], []
    for _, r in hits.iterrows():
        t = to_text(r.get("title")).strip()
        a = to_text(r.get("abstract")).strip()
        text = f"{t}. {a}".strip(". ").strip()

        parts = [s for s in re.split(r'(?<=[.!?])\s+', text) if len(s.split()) >= 6]
        if not parts:
            parts = [t or a or ""]  # at least one line

        sents  += parts
        src    += [to_text(r.get("url"))] * len(parts)
        titles += [t or a or "Article"] * len(parts)

    # TF-IDF relevance + simple MMR diversity
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", max_features=12000)
    mat = vec.fit_transform([query] + sents).toarray()
    qv, sv = mat[0], mat[1:]
    if sv.size == 0:
        return sents[0], []

    sim_q = (sv @ qv) / (np.linalg.norm(sv, axis=1) * (np.linalg.norm(qv) + 1e-12) + 1e-12)
    chosen = [int(sim_q.argmax())]
    while len(chosen) < min(max_sents, len(sents)):
        rest = [i for i in range(len(sents)) if i not in chosen]
        best, best_i = -1e9, None
        for i in rest:
            denom = (np.linalg.norm(sv[i]) + 1e-12)
            red = max((sv[i] @ sv[j]) / (denom * (np.linalg.norm(sv[j]) + 1e-12)) for j in chosen)
            score = 0.72 * sim_q[i] - 0.28 * red
            if score > best: best, best_i = score, i
        if best_i is None: break
        chosen.append(best_i)

    answer = " ".join([sents[i] for i in chosen])

    cites, seen = [], set()
    for i in chosen:
        u = src[i]
        if u and u not in seen:
            seen.add(u)
            cites.append((titles[i], u))
    return answer, cites


# 6) UI
st.title("🗞️ Personalized News Chatbot")
st.caption("Ask about topics. Answers are RAG summaries with citations; ranking blends your query with your profile (if available).")

with st.sidebar:
    st.subheader("Settings")
    user = st.selectbox("User profile", options=["(guest)"]+list(profiles.keys()),
                        index=0 if DEFAULT_USER is None else (list(profiles.keys()).index(DEFAULT_USER)+1))
    user = None if user=="(guest)" else user
    alpha = st.slider("α: query vs. taste", 0.0, 1.0, 0.6, 0.05)
    topk  = st.slider("Top-K docs", 1, 10, 5, 1)

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, content in st.session_state.chat:
    st.chat_message(role).markdown(content)

if prompt := st.chat_input("Type your question…"):
    st.session_state.chat.append(("user", prompt))
    st.chat_message("user").markdown(prompt)
    hits = retrieve_personalized(prompt, user_id=user, k=topk, alpha=alpha)
    answer, cites = summarize(prompt, hits, max_sents=5)
    who = user if user else "guest"
    msg = f"_profile: {who}_\n\n{answer}"
    if cites:
        msg += "\n\n**Sources**\n" + "\n".join(f"[{i+1}] {t}\n{u}" for i,(t,u) in enumerate(cites))
    st.session_state.chat.append(("assistant", msg))
    st.chat_message("assistant").markdown(msg)
# ------------------ end file ------------------
