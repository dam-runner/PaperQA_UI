import os
import logging
import json
import time
from time import sleep

from dotenv import load_dotenv

import streamlit as st
import requests

from paperqa import Settings, ask
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
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
# Silence internals
logging.getLogger("paperqa").setLevel(logging.INFO)
logging.getLogger("lite.llm").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

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

    settings.verbosity = 1 # Toggling verbosity setting

    return Settings(**common)

settings = build_settings()

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
        start = time.time()
        try:
            resp = ask(query, settings=settings)
            sess = resp.session  # type: PQASession

            # 1) Extract the answer text
            if sess.formatted_answer:
                ans = sess.formatted_answer
            else:
                ans = sess.answer or "No answer returned."

            # 2) Extract evidence snippets (contexts is a list[Context])
            evs = sess.contexts or []

            # 3) Extract cost directly from session
            #    (instead of computing from usage)
            cost = sess.cost

        except Exception as e:
            ans, evs, cost = f"Error: {e}", [], 0.0

        elapsed = time.time() - start

    # 4) Save to history
    st.session_state.history.append((query, ans, evs, elapsed, cost))

# — Display chat history —
for q, a, ctx_list, el, cst in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Answer ({el:.2f}s, Cost: ${cst:.4f}):**  {a}")

    # Show each snippet
    for c in ctx_list:
        snippet_text = c.text.text             # the actual snippet
        paper_doc    = c.text.doc              # a Doc object
        title        = paper_doc.title
        authors      = ", ".join(paper_doc.authors)
        citation_str = paper_doc.citation      # formatted bib entry
        page_num     = getattr(c, "page", None)
        link         = os.path.join(PAPERS_PATH, paper_doc.name)

        # Build a display name (must fall back gracefully)
        cite_display = f"{title} — {authors}" if authors else title
        if page_num is not None:
            cite_display += f" (p. {page_num})"

        st.markdown(f"> {snippet_text}\n> [📄 {cite_display}]({link})")


# ————————— Sidebar: Metrics —————————
st.sidebar.markdown("---")
st.sidebar.header("Session Metrics")
st.sidebar.markdown(f"**Total Cost:** ${st.session_state.total_cost:.4f}")

