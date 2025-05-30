import os
import logging
import shutil

# Configure logging to console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

import time
import json
import subprocess
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ratelimit import limits, sleep_and_retry
import time
import json
import subprocess
import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ratelimit import limits, sleep_and_retry

# ————————— Configuration —————————
from dotenv import load_dotenv  # load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "<YOUR_OPENAI_KEY>")
PAPERS_PATH     = os.getenv("PAPERS_PATH", "papers/")
# Normalize path separators
PAPERS_PATH = os.path.abspath(PAPERS_PATH)
MODEL_NAME      = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Verify PAPERS_PATH exists
if not os.path.isdir(PAPERS_PATH):
    raise FileNotFoundError(f"PAPERS_PATH not found: {PAPERS_PATH}")

# Pricing per token based on per-million rates based on per-million rates
PRICING = {
    "gpt-4o-mini":  {"prompt": 0.15/1e6, "completion": 0.60/1e6},
    "gpt-4.1-mini": {"prompt": 0.40/1e6, "completion": 1.60/1e6},
    "gpt-4.1-nano": {"prompt": 0.10/1e6, "completion": 0.40/1e6},
}

# ————————— Streamlit Setup —————————
st.set_page_config(page_title="PaperQA2 Chat", layout="wide")
st.title("📖 PaperQA2 Research Chat")

# ————————— Verify CLI Presence —————————
if shutil.which("pqa") is None:
    raise FileNotFoundError("Could not find 'pqa' CLI in PATH. Ensure paper-qa is installed and 'pqa' is available.")

# ————————— Hydration of Metadata —————————
# Load existing metadata or initialize empty map
data_file = "metadata.json"
if os.path.exists(data_file):
    with open(data_file, "r") as f:
        metadata_map = json.load(f)
else:
    metadata_map = {}

# One-time Crossref enrichment for each PDF
import requests
from time import sleep
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")
CROSSREF_URL    = "https://api.crossref.org/works"

def fetch_crossref_metadata(query_title: str) -> dict:
    params = {"query.title": query_title, "rows": 1}
    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO
    try:
        resp = requests.get(CROSSREF_URL, params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        if not items:
            return {}
        item = items[0]
        title = (item.get("title") or [""])[0]
        authors = [f"{a.get('given','')} {a.get('family','')}".strip()
                   for a in item.get("author", [])]
        year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
        return {"title": title, "authors": authors, "year": year}
    except Exception:
        return {}

# Iterate PDFs, enrich missing entries
for fname in os.listdir(PAPERS_PATH):
    if not fname.lower().endswith('.pdf'):
        continue
    if fname not in metadata_map or metadata_map[fname].get("year") is None:
        base = os.path.splitext(fname)[0]
        cr_data = fetch_crossref_metadata(base)
        if cr_data:
            metadata_map[fname] = cr_data
        else:
            metadata_map[fname] = {"title": base}
        # polite pause between requests
        sleep(1)

# Persist updated metadata
with open(data_file, "w") as f:
    json.dump(metadata_map, f, indent=2)

st.session_state.metadata_map = metadata_map

# ——————— CLI-based ask_query with rate-limit and retries ———————
@sleep_and_retry
@limits(calls=1, period=1)
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10), retry=retry_if_exception_type(Exception))
def ask_query(query: str, preset: str = None, custom: dict = None):
    """
    CLI invocation of 'pqa ask', returning JSON via --prompts.use_json.
    """
    # Build base CLI command: subcommand first, then flags
    cmd = ["pqa", "ask", "--prompts.use_json", "true"]
    if preset and preset != "Custom":
        cmd += ["-s", preset]
    elif custom:
        def flatten(d, prefix=""):
            for k, v in d.items():
                if isinstance(v, dict):
                    yield from flatten(v, prefix + k + ".")
                else:
                    yield prefix + k, v
        for key, val in flatten(custom):
            cmd += [f"--{key}", str(val)]
    # Append the user query
    cmd += [query]

    logger.debug(f"Running command: {' '.join(cmd)}")

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = OPENAI_API_KEY

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    except FileNotFoundError:
        logger.error("pqa CLI not found when attempting to run ask_query", exc_info=True)
        raise

    logger.debug(f"pqa stdout: {proc.stdout}")
    logger.debug(f"pqa stderr: {proc.stderr}")

    if proc.returncode != 0:
        logger.error(f"pqa command failed with return code {proc.returncode}")
        raise RuntimeError(proc.stderr.strip())

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON from pqa output", exc_info=True)
        raise

# ————————— Initialize Session State —————————
if "history" not in st.session_state:
    st.session_state.history = []  # stores (query, answer, evidence, elapsed, cost)
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0

# ————————— Sidebar: Presets & Settings —————————
st.sidebar.header("Retrieval Settings")

# Hard-code presets (having trouble grabbing them from CLI)
PRESETS = ["high_quality", "fast", "wikicrow", "contracrow", "debug", "Custom"]
selected = st.sidebar.selectbox("Preset Profile", PRESETS)

# Custom settings section
custom = {}
if selected == "Custom":
    temp     = st.sidebar.slider("Answer Temperature", 0.0, 1.0, 0.0, 0.05)
    sum_temp = st.sidebar.slider("RCS Temperature",    0.0, 1.0, 0.0, 0.05)
    k        = st.sidebar.number_input("Evidence k",    1, 50, 30)
    max_s    = st.sidebar.number_input("Max Sources",   1, 20, 5)
    detail   = st.sidebar.checkbox("Detailed Citations", True)
    sum_len  = st.sidebar.slider("Summary Length",    50, 500, 200, 50)
    csize    = st.sidebar.number_input("Chunk Size",   500, 5000, 2250, 250)
    ovlp     = st.sidebar.number_input("Overlap",        0, 1000, 200, 50)
    algo     = st.sidebar.selectbox("Chunk Algorithm", ["simple_overlap", "sections"])
    scnt     = st.sidebar.number_input("Search Count",    1, 5, 1)
    ret_meta = st.sidebar.checkbox("Return Metadata", True)
    custom = {
        "temperature": temp,
        "summary_temperature": sum_temp,
        "answer": {
            "evidence_k":                k,
            "evidence_detailed_citations": detail,
            "evidence_summary_length":   sum_len,
            "answer_max_sources":        max_s
        },
        "parsing": {
            "chunk_size":       csize,
            "overlap":          ovlp,
            "chunking_algorithm": algo
        },
        "agent": {
            "search_count":           scnt,
            "return_paper_metadata":  ret_meta
        }
    }

# ————————— Main Chat Interface —————————
query = st.text_input("Ask a question about your PDF library:")
if st.button("Send") and query:
    start = time.time()
    with st.spinner("Retrieving answer…"):
        try:
            if selected != "Custom":
                res = ask_query(query, preset=selected)
            else:
                res = ask_query(query, custom=custom)
            answer   = res.get("answer", "")
            evidence = res.get("evidence", [])
            usage    = res.get("usage", {})
        except Exception as e:
            answer, evidence, usage = f"Error: {e}", [], {}
    elapsed = time.time() - start

    # Cost calculation
    cost_text = "N/A"
    if usage and MODEL_NAME in PRICING:
        p, c = usage.get("prompt_tokens",0), usage.get("completion_tokens",0)
        rate  = PRICING[MODEL_NAME]
        val   = p*rate["prompt"] + c*rate["completion"]
        st.session_state.total_cost += val
        cost_text = f"${val:.4f}"

    st.session_state.history.append((query, answer, evidence, elapsed, cost_text))

# ————————— Display Chat History —————————
for q, a, evs, el, cst in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Answer ({el:.2f}s, Cost: {cst}):** {a}")
    for ev in evs:
        text = ev.get("text", "")
        pid  = ev.get("paper_id", "Unknown.pdf")
        pg   = ev.get("page")
        md   = st.session_state.metadata_map.get(pid, {})
        title   = md.get("title", pid)
        authors = ", ".join(md.get("authors", []))
        link    = os.path.join(PAPERS_PATH, pid)
        cite    = f"{title} — {authors}" if authors else title
        if pg is not None:
            cite += f" (p. {pg})"
        st.markdown(f"> {text}\n> [📄 {cite}]({link})")

# ————————— Sidebar: Session Metrics —————————
st.sidebar.markdown("---")
st.sidebar.header("Session Metrics")
st.sidebar.markdown(f"**Total Cost:** ${st.session_state.total_cost:.4f}")

