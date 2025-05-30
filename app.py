import os
import time
import json
import streamlit as st
from paperqa import PaperQAClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ratelimit import limits, sleep_and_retry

# ————————— Configuration —————————
# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "<YOUR_OPENAI_KEY>")
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")  # included if needed
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
PAPERS_PATH = os.getenv("PAPERS_PATH", "papers/")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Token pricing per million tokens converted to per-token rates
PRICING = {
    "gpt-4o-mini": {"prompt": 0.15 / 1_000_000, "completion": 0.60 / 1_000_000},
    "gpt-4.1-mini": {"prompt": 0.40 / 1_000_000, "completion": 1.60 / 1_000_000},
    "gpt-4.1-nano": {"prompt": 0.10 / 1_000_000, "completion": 0.40 / 1_000_000},
}

# ————————— Streamlit Setup —————————
st.set_page_config(page_title="📖 PaperQA2 Chat", layout="wide")
st.title("📖 PaperQA2 Research Chat")

# ————————— Initialize Client —————————
if "pqa_client" not in st.session_state:
    # PaperQAClient instantiation without explicit mailto; Crossref handled internally
    st.session_state.pqa_client = PaperQAClient(
        papers_path=PAPERS_PATH,
        openai_api_key=OPENAI_API_KEY,
        semantic_scholar_api_key=SEMANTIC_SCHOLAR_API_KEY
    )
client = st.session_state.pqa_client

# ————————— Hydration of Metadata —————————
# metadata.json generation or loading to supply human-readable citations
metadata_file = "metadata.json"
if os.path.exists(metadata_file):
    with open(metadata_file, "r") as f:
        metadata_map = json.load(f)
else:
    metadata_map = {}
    paper_ids = client.list_papers()
    for pid in paper_ids:
        try:
            info = client.get_metadata(pid)
        except Exception:
            info = {"title": pid}
        metadata_map[pid] = info
    with open(metadata_file, "w") as f:
        json.dump(metadata_map, f, indent=2)
# store for session use
st.session_state.metadata_map = metadata_map

# ——————— Rate-limited, Retrying Ask Function ———————
# maximum 1 call per second, retries on exception up to 5 attempts with exponential backoff
@sleep_and_retry
@limits(calls=1, period=1)
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type(Exception))
def ask_query(query: str, **kwargs):
    return client.ask(query, **kwargs)

# ————————— Initialize Session State —————————
if "history" not in st.session_state:
    st.session_state.history = []  # (query, answer, evidence, elapsed, cost)
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# ————————— Sidebar: Presets & Settings —————————
st.sidebar.header("Retrieval Settings")
try:
    preset_list = client.list_settings()
except Exception:
    preset_list = ["high_quality", "fast", "wikicrow", "contracrow", "debug"]
preset_list.append("Custom")
selected_preset = st.sidebar.selectbox("Preset Profile", preset_list)

custom_settings = {}
if selected_preset == "Custom":
    temp        = st.sidebar.slider("Answer Temperature", 0.0, 1.0, 0.0, 0.05)
    sum_temp    = st.sidebar.slider("RCS Temperature",    0.0, 1.0, 0.0, 0.05)
    k           = st.sidebar.number_input("Evidence k (RCS chunks)", 1, 50, 30)
    max_src     = st.sidebar.number_input("Max Sources", 1, 20, 5)
    detailed    = st.sidebar.checkbox("Detailed Citations", value=True)
    sum_len     = st.sidebar.slider("Summary Length", 50, 500, 200, 50)
    chunk_size  = st.sidebar.number_input("Chunk Size", 500, 5000, 2250, 250)
    overlap     = st.sidebar.number_input("Overlap", 0, 1000, 200, 50)
    chunk_algo  = st.sidebar.selectbox("Chunk Algorithm", ["simple_overlap", "sections"])
    search_cnt  = st.sidebar.number_input("Agent Search Count", 1, 5, 1)
    return_meta = st.sidebar.checkbox("Return Paper Metadata", value=True)
    custom_settings = {
        "temperature":         temp,
        "summary_temperature": sum_temp,
        "answer": {
            "evidence_k":                k,
            "evidence_detailed_citations": detailed,
            "evidence_summary_length":   sum_len,
            "answer_max_sources":        max_src
        },
        "parsing": {
            "chunk_size":       chunk_size,
            "overlap":          overlap,
            "chunking_algorithm": chunk_algo
        },
        "agent": {
            "search_count":           search_cnt,
            "return_paper_metadata":  return_meta
        }
    }

# ————————— Main Chat Interface —————————
query = st.text_input("Ask a question about your PDF library:")
if st.button("Send") and query:
    start_time = time.time()
    with st.spinner("⏳ Retrieving answer…"):
        try:
            if selected_preset != "Custom":
                resp = ask_query(query, preset=selected_preset)
            else:
                resp = ask_query(query, settings=custom_settings)
            answer   = resp.get("answer", "No answer returned.")
            evidence = resp.get("evidence", [])
            usage    = resp.get("usage", {})
        except Exception as e:
            answer, evidence, usage = f"Error: {e}", [], {}
    elapsed = time.time() - start_time

    # cost calculation with default fallback
    cost_text = "No Pricing Info Available"
    if usage and MODEL_NAME in PRICING:
        ptoks = usage.get("prompt_tokens", 0)
        ctoks = usage.get("completion_tokens", 0)
        rates = PRICING[MODEL_NAME]
        cost_val = ptoks * rates["prompt"] + ctoks * rates["completion"]
        st.session_state.total_cost += cost_val
        cost_text = f"${cost_val:.4f}"

    st.session_state.history.append((query, answer, evidence, elapsed, cost_text))

# ————————— Display Chat History —————————
for q, a, ev_list, el, cost in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Answer ({el:.2f}s, Cost: {cost}):** {a}")
    for ev in ev_list:
        text    = ev.get("text", "")
        pid     = ev.get("paper_id", "Unknown.pdf")
        page    = ev.get("page")
        info    = st.session_state.metadata_map.get(pid, {})
        title   = info.get("title", pid)
        authors = ", ".join(info.get("authors", []))
        link    = os.path.join(PAPERS_PATH, pid)
        citation = f"{title} — {authors}" if authors else title
        if page is not None:
            citation += f" (p. {page})"
        st.markdown(f"> {text}\n> [📄 {citation}]({link})")

# ————————— Sidebar: Session Metrics —————————
st.sidebar.markdown("---")
st.sidebar.header("Session Metrics")
st.sidebar.markdown(f"**Total Cost:** ${st.session_state.total_cost:.4f}")

