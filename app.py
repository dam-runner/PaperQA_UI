import os
import logging
from dotenv import load_dotenv

import streamlit as st
import requests
from time import sleep

from paperqa import Settings, ask, index
from pydantic import BaseModel

# ————————— Configuration —————————
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PAPERS_PATH   = os.path.abspath(os.getenv("PAPERS_PATH", "papers/"))

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")
if not os.path.isdir(PAPERS_PATH):
    raise FileNotFoundError(f"PAPERS_PATH not found: {PAPERS_PATH}")

# Pricing (per-token rates)
PRICING = {
    "gpt-4o-mini":  {"prompt": 0.15/1e6, "completion": 0.60/1e6},
    "gpt-4.1-mini": {"prompt": 0.40/1e6, "completion": 1.60/1e6},
    "gpt-4.1-nano": {"prompt": 0.10/1e6, "completion": 0.40/1e6},
}

# ————————— Logging —————————
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# ————————— Streamlit Setup —————————
st.set_page_config(page_title="PaperQA2 Chat", layout="wide")
st.title("📖 PaperQA2 Research Chat")

# ————————— Sidebar: Settings —————————
st.sidebar.header("Retrieval Settings")
PRESETS = ["high_quality", "fast", "wikicrow", "contracrow", "debug", "Custom"]
selected = st.sidebar.selectbox("Preset Profile", PRESETS)

def build_settings():
    # Common settings
    common = {
        "openai_api_key": OPENAI_API_KEY,
        "paper_directory": PAPERS_PATH
    }
    if selected != "Custom":
        return Settings(**common, setting_name=selected)
    # Custom sliders
    custom = {
        "temperature": st.sidebar.slider("Answer Temp", 0.0, 1.0, 0.0, 0.05),
        "summary_temperature": st.sidebar.slider("RCS Temp", 0.0, 1.0, 0.0, 0.05),
        "parsing": {
            "chunk_size":      st.sidebar.number_input("Chunk Size", 500, 5000, 2250, 250),
            "overlap":         st.sidebar.number_input("Overlap",    0, 1000, 200, 50),
            "chunking_algorithm": st.sidebar.selectbox("Chunk Algo", ["simple_overlap","sections"])
        },
        "answer": {
            "evidence_k":                st.sidebar.number_input("Evidence k",1,50,30),
            "answer_max_sources":        st.sidebar.number_input("Max Sources",1,20,5),
            "evidence_detailed_citations": st.sidebar.checkbox("Detailed Citations", True),
            "evidence_summary_length":   st.sidebar.slider("Summary Length",50,500,200,50)
        },
        "agent": {
            "search_count":            st.sidebar.number_input("Search Count",1,5,1),
            "return_paper_metadata":   st.sidebar.checkbox("Return Metadata", True)
        }
    }
    common.update(custom)
    return Settings(**common)

settings = build_settings()

# ————————— Indexing (cached) —————————
@st.cache_resource(show_spinner=False)
def ensure_index(path: str):
    logger.info("Ensuring PaperQA index exists…")
    index(paper_directory=path)  # idempotent
    return True

_ = ensure_index(PAPERS_PATH)

# ————————— Metadata Enrichment (cached) —————————
@st.cache_data(ttl=24*3600)
def load_metadata(papers_dir: str):
    meta_path = "metadata.json"
    if os.path.exists(meta_path):
        return json.load(open(meta_path))
    # one-time Crossref enrichment
    out = {}
    for fname in os.listdir(papers_dir):
        if not fname.endswith(".pdf"):
            continue
        title_guess = os.path.splitext(fname)[0]
        params = {"query.title": title_guess, "rows": 1}
        mailto = os.getenv("CROSSREF_MAILTO")
        if mailto:
            params["mailto"] = mailto
        try:
            r = requests.get("https://api.crossref.org/works", params=params, timeout=10)
            r.raise_for_status()
            items = r.json().get("message",{}).get("items",[])
            if items:
                item = items[0]
                out[fname] = {
                    "title": (item.get("title") or [""])[0],
                    "authors": [f"{a.get('given','')} {a.get('family','')}".strip()
                                for a in item.get("author",[])],
                    "year": item.get("issued",{}).get("date-parts",[[None]])[0][0]
                }
            else:
                out[fname] = {"title": title_guess}
        except Exception:
            out[fname] = {"title": title_guess}
        sleep(1)
    with open(meta_path, "w") as f:
        json.dump(out, f, indent=2)
    return out

metadata_map = load_metadata(PAPERS_PATH)
st.session_state.metadata_map = metadata_map

# ————————— Chat Interface —————————
if "history" not in st.session_state:
    st.session_state.history = []
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

query = st.text_input("Ask a question about your PDF library:")
if st.button("Send") and query:
    with st.spinner("Retrieving answer…"):
        start = st.time()
        try:
            resp = ask(query, settings=settings)
            ans  = resp.formatted_answer
            evs  = resp.context
            usage = resp.usage or {}
        except Exception as e:
            ans, evs, usage = f"Error: {e}", [], {}
        elapsed = st.time() - start

    # Cost tracking
    cost = 0.0
    model = settings.llm_config.get("model_name", "")
    if usage and model in PRICING:
        p = usage.get("prompt_tokens",0)
        c = usage.get("completion_tokens",0)
        rate = PRICING[model]
        cost = p*rate["prompt"] + c*rate["completion"]
        st.session_state.total_cost += cost

    st.session_state.history.append((query, ans, evs, elapsed, cost))

# ————————— Display History —————————
for q,a,evs,el,cst in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Answer** ({el:.2f}s, Cost: ${cst:.4f}):  {a}")
    for ev in evs:
        text = ev.text
        pid  = ev.paper_id
        pg   = ev.page
        meta = metadata_map.get(pid,{})
        cite = f"{meta.get('title',pid)} — {', '.join(meta.get('authors',[]))}"
        if pg: cite += f" (p. {pg})"
        link = os.path.join(PAPERS_PATH, pid)
        st.markdown(f"> {text}\n> [📄 {cite}]({link})")

# ————————— Sidebar: Metrics —————————
st.sidebar.markdown("---")
st.sidebar.header("Session Metrics")
st.sidebar.markdown(f"**Total Cost:** ${st.session_state.total_cost:.4f}")

