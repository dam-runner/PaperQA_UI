import os
import time
import json
import streamlit as st
import asyncio

# Async client entrypoints and settings
from paperqa.clients.settings import Settings
from paperqa.clients.metadata import get_directory_index
from paperqa.clients.agent import agent_query

# Retry & rate-limit tools
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ratelimit import limits, sleep_and_retry

# ————————— Configuration —————————
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "<YOUR_OPENAI_KEY>")
PAPERS_PATH     = os.getenv("PAPERS_PATH", "papers/")
MODEL_NAME      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PRICING = {
    "gpt-4o-mini":  {"prompt": 0.15/1e6, "completion": 0.60/1e6},
    "gpt-4.1-mini": {"prompt": 0.40/1e6, "completion": 1.60/1e6},
    "gpt-4.1-nano": {"prompt": 0.10/1e6, "completion": 0.40/1e6},
}

# ————————— Streamlit Setup —————————
st.set_page_config(page_title="PaperQA2 Chat", layout="wide")
st.title("📖 PaperQA2 Research Chat")

# ————————— Initialize PaperQA Settings & Index —————————
# Settings object configures paths & API keys
if "pqa_settings" not in st.session_state:
    st.session_state.pqa_settings = Settings(
        paper_directory=PAPERS_PATH,
        openai_api_key=OPENAI_API_KEY
    )
    # Build the directory index once (metadata + caching)
    asyncio.run(get_directory_index(settings=st.session_state.pqa_settings))

settings = st.session_state.pqa_settings

# ————————— Hydrate Our Own metadata.json —————————
metadata_file = "metadata.json"
if os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        metadata_map = json.load(f)
else:
    metadata_map = {}
    # Pull each paper’s metadata via async client
    paper_ids = settings._paper_ids  # internal list from get_directory_index
    for pid in paper_ids:
        try:
            md = asyncio.run(settings.get_metadata(pid))
        except Exception:
            md = {"title": pid}
        metadata_map[pid] = md
    with open(metadata_file, "w") as f:
        json.dump(metadata_map, f, indent=2)

st.session_state.metadata_map = metadata_map

# ——————— Rate-limited, Retrying Ask Function ———————
@sleep_and_retry
@limits(calls=1, period=1)
@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=1, max=10),
       retry=retry_if_exception_type(Exception))
def ask_query(query: str, preset: str = None, custom: dict = None):
    # Wrap the async agent_query in asyncio.run
    return asyncio.run(
        agent_query(
            query=query,
            settings=settings,
            preset=preset,
            custom_settings=custom
        )
    )

# ————————— Initialize Session State —————————
if "history" not in st.session_state:
    st.session_state.history = []
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# ————————— Sidebar: Presets & Settings —————————
st.sidebar.header("Retrieval Settings")
try:
    builtins = settings.list_presets()
except Exception:
    builtins = ["high_quality", "fast", "wikicrow", "contracrow", "debug"]
builtins.append("Custom")
selected = st.sidebar.selectbox("Preset Profile", builtins)

custom = {}
if selected == "Custom":
    # LLM
    temp     = st.sidebar.slider("Answer Temperature", 0.0,1.0,0.0,0.05)
    sum_temp = st.sidebar.slider("RCS Temperature",    0.0,1.0,0.0,0.05)
    # Evidence
    k        = st.sidebar.number_input("Evidence k",      1,50,30)
    max_s    = st.sidebar.number_input("Max Sources",     1,20,5)
    detail   = st.sidebar.checkbox("Detailed Citations", True)
    sum_len  = st.sidebar.slider("Summary Length",      50,500,200,50)
    # Parsing
    csize    = st.sidebar.number_input("Chunk Size",   500,5000,2250,250)
    ovlp     = st.sidebar.number_input("Overlap",        0,1000,200,50)
    algo     = st.sidebar.selectbox("Chunk Algorithm", ["simple_overlap","sections"])
    # Agent
    scnt     = st.sidebar.number_input("Search Count",    1,5,1)
    ret_meta = st.sidebar.checkbox("Return Metadata", True)

    custom = {
        "temperature": temp,
        "summary_temperature": sum_temp,
        "answer": {
            "evidence_k":                k,
            "evidence_detailed_citations": detail,
            "evidence_summary_length":   sum_len,
            "answer_max_sources":        max_s,
        },
        "parsing": {
            "chunk_size":       csize,
            "overlap":          ovlp,
            "chunking_algorithm": algo,
        },
        "agent": {
            "search_count":          scnt,
            "return_paper_metadata": ret_meta,
        }
    }

# ————————— Main Chat Interface —————————
query = st.text_input("Ask a question about your PDF library:")
if st.button("Send") and query:
    start = time.time()
    with st.spinner("Retrieving…"):
        try:
            if selected != "Custom":
                res = ask_query(query, preset=selected)
            else:
                res = ask_query(query, custom=custom)
            answer   = res["answer"]
            evidence = res["evidence"]
            usage    = res["usage"]
        except Exception as e:
            answer, evidence, usage = f"Error: {e}", [], {}

    elapsed = time.time() - start

    # Compute cost
    cost = "N/A"
    if usage and MODEL_NAME in PRICING:
        p,c = usage.get("prompt_tokens",0), usage.get("completion_tokens",0)
        rate = PRICING[MODEL_NAME]
        val  = p*rate["prompt"] + c*rate["completion"]
        st.session_state.total_cost += val
        cost = f"${val:.4f}"

    st.session_state.history.append((query, answer, evidence, elapsed, cost))

# ————————— Display Chat History —————————
for q,a,evs,el,cst in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Answer ({el:.2f}s, Cost: {cst}):** {a}")
    for ev in evs:
        txt = ev["text"]
        pid = ev["paper_id"]
        pg  = ev.get("page")
        md  = st.session_state.metadata_map.get(pid, {})
        title   = md.get("title", pid)
        authors = ", ".join(md.get("authors", []))
        link    = os.path.join(PAPERS_PATH, pid)
        cite    = f"{title} — {authors}" if authors else title
        if pg is not None:
            cite += f" (p. {pg})"
        st.markdown(f"> {txt}\n> [📄 {cite}]({link})")

# ————————— Sidebar: Session Metrics —————————
st.sidebar.markdown("---")
st.sidebar.header("Session Metrics")
st.sidebar.markdown(f"**Total Cost:** ${st.session_state.total_cost:.4f}")

