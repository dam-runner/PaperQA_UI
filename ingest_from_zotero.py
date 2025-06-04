#!/usr/bin/env python3
"""
robust_ingest_from_zotero.py

A robust metadata ingestion pipeline from a Zotero CSV, using:
  - arXiv ID lookup (if Zotero URL is an arXiv link)
  - Zotero’s DOI (if present and returns data)
  - CrossRef fuzzy-title → new DOI → query
  - Fallback: title + authors with high fuzzy match threshold

All mismatch or failure events are logged to mismatch.log.
Final output is a fully JSON-safe metadata_trimmed.json.

Usage:
  1. Ensure .env contains at least:
       OPENAI_API_KEY=sk-...
       (Optional) CROSSREF_MAILTO=you@example.com
       (Optional) SEMANTIC_SCHOLAR_API_KEY=...
  2. Install dependencies:
       pip install paperqa aiohttp python-dotenv requests rapidfuzz
  3. Place zotero.csv in the same directory.
  4. Run:
       python robust_ingest_from_zotero.py
"""

import os
import csv
import json
import re
import asyncio
import requests
import datetime
import aiohttp
from dotenv import load_dotenv
from rapidfuzz import fuzz
from paperqa.clients import DocMetadataClient, ALL_CLIENTS

# ─────────────────────────────────────────────────────────────────────────────
# 0) Load environment variables
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", None)
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", None)
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", None)

if not OPENAI_API_KEY:
    raise RuntimeError("⚠️ Missing OPENAI_API_KEY in .env")

# ─────────────────────────────────────────────────────────────────────────────
# 1) Helper functions
# ─────────────────────────────────────────────────────────────────────────────

ARXIV_PATTERN = re.compile(r"^https?://arxiv\.org/abs/([^/]+)")

def extract_arxiv_id(url: str) -> str | None:
    """
    If url matches 'http://arxiv.org/abs/<id>' or 'https://arxiv.org/abs/<id>',
    return the <id> part, else None.
    """
    m = ARXIV_PATTERN.match(url or "")
    return m.group(1) if m else None

def clean_title(raw: str) -> str:
    """
    Simplify raw title by:
      1) Stripping anything after first colon, em-dash, or parenthesis.
      2) Removing remaining punctuation.
    """
    core = re.split(r"[:—\(]", raw)[0].strip()
    core = re.sub(r"[^\w\s]", "", core)
    return core

def find_doi_via_crossref(title_guess: str, first_author: str | None = None) -> str | None:
    """
    Given a title guess (shortened), query CrossRef (/works?query.title=…) for the top hit.
    Return its DOI string if found, else None.
    """
    params = {"query.title": title_guess, "rows": 1}
    if first_author:
        params["query.author"] = first_author
    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO
    try:
        resp = requests.get("https://api.crossref.org/works", params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
        if items:
            return items[0].get("DOI")
    except Exception:
        pass
    return None

def make_json_safe(obj):
    """
    Recursively convert any datetime.datetime → ISO string,
    set/tuple → list, and sanitize nested dict/list.
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, (set, tuple)):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    return obj  # str, int, float, bool, None are JSON-safe

# ── New helper: direct arXiv HTTP fetch ──
async def fetch_arxiv_direct(arxiv_id: str) -> dict | None:
    """
    If PaperQA’s arXiv lookup failed, call the official arXiv API directly
    and return a dict with keys:
      "title", "authors", "year", "abstract", "pdf_url".
    Return None if the HTTP lookup doesn’t yield an <entry>.
    """
    api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(api_url, timeout=10) as resp:
            if resp.status != 200:
                return None
            text = await resp.text()
    # Now parse the XML. arXiv’s response always wraps each paper in <entry>…</entry>.
    # A minimal parse (without bringing in feedparser) is:
    m = re.search(r"<entry>(.*?)</entry>", text, re.DOTALL)
    if not m:
        return None

    entry_xml = m.group(1)
    # Extract title
    title_m = re.search(r"<title>(.*?)</title>", entry_xml, re.DOTALL)
    title = title_m.group(1).strip() if title_m else None

    # Extract authors (there are multiple <author><name>…</name></author>)
    authors = [_.group(1).strip() for _ in re.finditer(r"<author>\s*<name>(.*?)</name>", entry_xml, re.DOTALL)]

    # Extract published date (e.g. 2021-04-05T…)
    pub_m = re.search(r"<published>(\d{4})-(\d{2})-(\d{2})T", entry_xml)
    year = int(pub_m.group(1)) if pub_m else None

    # Extract abstract (<summary>…</summary>)
    abs_m = re.search(r"<summary>(.*?)</summary>", entry_xml, re.DOTALL)
    abstract = abs_m.group(1).strip() if abs_m else None

    # Extract the PDF link (there is a link tag with rel="related" and title="pdf", but easier: rel="alternate" for HTML, and rel="related" type="application/pdf")
    pdf_url = None
    for link_m in re.finditer(r'<link\s+rel="related"\s+type="application/pdf"\s+href="(.*?)"', entry_xml):
        pdf_url = link_m.group(1)
        break

    return {
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
        "pdf_url": pdf_url
    }

# ─────────────────────────────────────────────────────────────────────────────
# 2) Load Zotero CSV
# ─────────────────────────────────────────────────────────────────────────────
ZOTERO_CSV = "zotero.csv"
all_meta: dict[str, dict] = {}

with open(ZOTERO_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    # Print fieldnames for debugging
    print("Zotero CSV Headers:", reader.fieldnames)

    for row in reader:
        raw_title = row.get("Title", "").strip()
        if not raw_title:
            continue

        # Use the exact Zotero Title as the key in our dictionary
        fname = raw_title

        authors = [a.strip() for a in row.get("Author", "").split(";") if a.strip()]
        doi_zotero = row.get("DOI", "").strip() or None
        url = row.get("Url", "").strip() or None
        journal = row.get("Publication Title", "").strip() or None
        year = None
        if row.get("Publication Year", "").isdigit():
            year = int(row["Publication Year"])
        genre = row.get("Item Type", "").strip() or None
        link_attachments = [
            l.strip() for l in row.get("Link Attachments", "").split(";") if l.strip()
        ]
        manual_tags = [
            t.strip() for t in row.get("Manual Tags", "").split(";") if t.strip()
        ]
        automatic_tags = [
            t.strip() for t in row.get("Automatic Tags", "").split(";") if t.strip()
        ]
        pages = row.get("Pages", "").strip() or None
        issue = row.get("Issue", "").strip() or None
        volume = row.get("Volume", "").strip() or None
        publisher = row.get("Publisher", "").strip() or None
        abstract = row.get("Abstract Note", "").strip() or None

        # Initialize with Zotero-provided fields; API fields start as None
        all_meta[fname] = {
            "title": raw_title,
            "authors": authors or None,
            "doi": doi_zotero,
            "url": url,
            "journal": journal,
            "year": year,
            "genre": genre,
            "link_attachments": link_attachments,
            "manual_tags": manual_tags,
            "automatic_tags": automatic_tags,
            "pages": pages,
            "issue": issue,
            "volume": volume,
            "publisher": publisher,
            "abstract": abstract,
            # Placeholders for API enrichments:
            "publication_date": None,
            "citation_count": None,
            "reference_count": None,
            "is_open_access": None,
            "pdf_url": None,
            "container_title": None,
            "fields_of_study": None,
            "crossref_subjects": None,
            "subject": None,
            "bibtex_source": [],
            "source_quality": None,
            "arxiv_id": None,
        }

# ─────────────────────────────────────────────────────────────────────────────
# 3) Load manual_overrides.json (if present)
# ─────────────────────────────────────────────────────────────────────────────
manual_overrides = {}
if os.path.exists("manual_overrides.json"):
    with open("manual_overrides.json", "r", encoding="utf-8") as f:
        manual_overrides = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Asynchronous “fill entry” coroutine
# ─────────────────────────────────────────────────────────────────────────────
sem = asyncio.Semaphore(1)

async def fill_entry(fname: str, entry: dict) -> dict:
    """
    1. Try manual override
    2. Try arXiv (PaperQA → if that fails, direct HTTP)
    3. Try Zotero DOI (but only accept if title fuzz ≥ 90)
    4. Try CrossRef fuzzy‐title → DOI (top 3 hits, token_sort_ratio ≥ 93, then verify returned title ≥ 90)
    5. Try title+author fallback (token_sort_ratio ≥ 90, same year, ≥ 50% last‐name overlap)
    6. If none succeeded, log exactly one line to mismatch.log, and copy Zotero link_attachments→pdf_url if present.
    7. Finally, drop the eight unwanted keys before returning.
    """

    client = DocMetadataClient(clients=ALL_CLIENTS)
    raw_title = entry.get("title", "").strip()
    authors = entry.get("authors", []) or []
    zotero_year = entry.get("year")  # may be None
    merged_any = False

    # ───── 0) Manual override ─────
    if fname in manual_overrides:
        entry.update(manual_overrides[fname])
        # We consider this “merged” even if manual override only sets some fields
        merged_any = True
        # Drop unwanted keys (see step 7) before returning
        for bad_key in [
            "bibtex_source", "arxiv_id", "docname", "dockey",
            "fields_to_overwrite_from_metadata", "key",
            "bibtex_type", "doc_id"
        ]:
            entry.pop(bad_key, None)
        return make_json_safe(entry)

    # ───── 1) STEP 0: Try arXiv via PaperQA ─────
    arxiv_id = None
    # If Zotero provided a URL like "https://arxiv.org/abs/XXXX.XXXXX"
    url = entry.get("url", "") or ""
    m = re.search(r"arxiv\.org/abs/([0-9]+\.[0-9]+)", url)
    if m:
        arxiv_id = m.group(1)

    if arxiv_id:
        entry["arxiv_id"] = arxiv_id
        # 1a) PaperQA lookup
        async with sem:
            await asyncio.sleep(1.0)  # preserve your existing rate‐limit
            try:
                details = await client.query(
                    title=cleaned,
                    authors=entry.get("authors", []),
                    fields=[
                        "title", "authors", "year", "doi", "citation_count",
                        "reference_count", "is_open_access", "pdf_url",
                        "fields_of_study", "crossref_subjects", "publisher",
                        "container_title", "bibtex_source", "source_quality",
                    ],
                )
            except:
                details = None

        if details:
            md = details.model_dump()
            returned_title = (md.get("title") or "").strip()
            # Only accept if fuzzy match to raw_title ≥ 90
            sim = fuzz.token_sort_ratio(raw_title.lower(), returned_title.lower())
            if sim >= 90:
                for k, v in md.items():
                    if v is not None:
                        entry[k] = v
                merged_any = True
                # drop unwanted keys, then return
                for bad_key in [
                    "bibtex_source", "docname", "dockey",
                    "fields_to_overwrite_from_metadata", "key",
                    "bibtex_type", "doc_id"
                ]:
                    entry.pop(bad_key, None)
                return make_json_safe(entry)
            # else: PaperQA had an arXiv record but it returned the “wrong” title.
            # fall through to HTTP fallback

        # 1b) Direct HTTP fallback to arXiv API
        http_meta = await fetch_arxiv_direct(arxiv_id)
        if http_meta:
            # http_meta contains at least {title, authors, year, abstract, pdf_url}
            for k, v in http_meta.items():
                if v is not None:
                    entry[k] = v
            merged_any = True
            # drop unwanted keys, then return
            for bad_key in [
                "bibtex_source", "arxiv_id", "docname", "dockey",
                "fields_to_overwrite_from_metadata", "key",
                "bibtex_type", "doc_id"
            ]:
                entry.pop(bad_key, None)
            return make_json_safe(entry)

        # If we reach here, both PaperQA and HTTP arXiv failed → do NOT log yet.
        # fall through to STEP 1

    # ───── 2) STEP 1: Zotero DOI (if present) ─────
    zotero_doi = entry.get("doi")
    if zotero_doi:
        async with sem:
            await asyncio.sleep(1.0)
            try:
                details = await client.query(
                    title=cleaned,
                    authors=entry.get("authors", []),
                    fields=[
                        "title", "authors", "year", "doi", "citation_count",
                        "reference_count", "is_open_access", "pdf_url",
                        "fields_of_study", "crossref_subjects", "publisher",
                        "container_title", "bibtex_source", "source_quality",
                    ],
                )
            except:
                details = None

        if details:
            md = details.model_dump()
            returned_title = (md.get("title") or "").strip()
            sim = fuzz.token_sort_ratio(raw_title.lower(), returned_title.lower())
            if sim >= 90:
                for k, v in md.items():
                    if v is not None:
                        entry[k] = v
                merged_any = True
                # drop unwanted keys, then return
                for bad_key in [
                    "bibtex_source", "arxiv_id", "docname", "dockey",
                    "fields_to_overwrite_from_metadata", "key",
                    "bibtex_type", "doc_id"
                ]:
                    entry.pop(bad_key, None)
                return make_json_safe(entry)
            # else: Zotero’s DOI returned a mismatched title → fall through

    # ───── 3) STEP 2: CrossRef fuzzy‐title → DOI ─────
    # Build two query strings: raw_title (with punctuation) and stripped (no punctuation)
    stripped = re.sub(r"[^\w\s]", "", raw_title)
    params = {
        "query.title": raw_title,
        "rows": 3
    }
    if authors:
        # Use last name of first author to narrow CrossRef search
        params["query.author"] = authors[0].split()[-1]

    try:
        resp = requests.get("https://api.crossref.org/works", params=params, timeout=10)
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])[:3]
    except:
        items = []

    best_candidate = None
    best_score = 0
    for candidate in items:
        cand_title = candidate.get("title", [""])[0]
        score = fuzz.token_sort_ratio(stripped.lower(), cand_title.lower())
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_score >= 93 and best_candidate:
        found_doi = best_candidate.get("DOI")
        if found_doi:
            entry["doi"] = found_doi
            async with sem:
                await asyncio.sleep(1.0)
                try:
                    details = await client.query(
                        title=cleaned,
                        authors=entry.get("authors", []),
                        fields=[
                            "title", "authors", "year", "doi", "citation_count",
                            "reference_count", "is_open_access", "pdf_url",
                            "fields_of_study", "crossref_subjects", "publisher",
                            "container_title", "bibtex_source", "source_quality",
                        ],
                    )
                except:
                    details = None

            if details:
                md = details.model_dump()
                returned_title = (md.get("title") or "").strip()
                sim = fuzz.token_sort_ratio(raw_title.lower(), returned_title.lower())
                if sim >= 90:
                    for k, v in md.items():
                        if v is not None:
                            entry[k] = v
                    merged_any = True
                    # drop unwanted keys, then return
                    for bad_key in [
                        "bibtex_source", "arxiv_id", "docname", "dockey",
                        "fields_to_overwrite_from_metadata", "key",
                        "bibtex_type", "doc_id"
                    ]:
                        entry.pop(bad_key, None)
                    return make_json_safe(entry)
            # else: CrossRef’s DOI returned a mismatched title → fall through

    # ───── 4) STEP 3: title+author fallback ─────
    if not entry.get("doi"):
        cleaned = re.sub(r"[^\w\s]", "", raw_title)
        fallback_kwargs: dict[str, object] = {"title": cleaned, "authors": authors}
        if zotero_year:
            fallback_kwargs["year"] = zotero_year

        async with sem:
            await asyncio.sleep(1.0)
            try:
                details = await client.query(
                    title=cleaned,
                    authors=entry.get("authors", []),
                    fields=[
                        "title", "authors", "year", "doi", "citation_count",
                        "reference_count", "is_open_access", "pdf_url",
                        "fields_of_study", "crossref_subjects", "publisher",
                        "container_title", "bibtex_source", "source_quality",
                    ],
                )
            except:
                details = None

        if details:
            md = details.model_dump()
            ret_title = (md.get("title") or "").strip().lower()
            sim_title = fuzz.token_sort_ratio(raw_title.lower(), ret_title)
            ret_year = md.get("year")
            year_ok = (zotero_year is not None and ret_year == zotero_year)

            # Check author last‐name overlap
            zotero_last = {a.split()[-1].lower() for a in authors}
            ret_last = {a.split()[-1].lower() for a in (md.get("authors") or [])}
            overlap = len(zotero_last & ret_last) / max(len(zotero_last), 1) if zotero_last else 0

            if sim_title >= 90 and year_ok and overlap >= 0.5:
                for k, v in md.items():
                    if v is not None:
                        entry[k] = v
                merged_any = True
                # drop unwanted keys, then return
                for bad_key in [
                    "bibtex_source", "arxiv_id", "docname", "dockey",
                    "fields_to_overwrite_from_metadata", "key",
                    "bibtex_type", "doc_id"
                ]:
                    entry.pop(bad_key, None)
                return make_json_safe(entry)
            # else: fallback returned something, but it didn’t meet our (title+year+author) criteria

    # ───── 5) STEP 4: Final fallback “give up” ─────
    if not merged_any:
        # If Zotero had a link attachment, use it as pdf_url if pdf_url is still empty
        if not entry.get("pdf_url") and entry.get("link_attachments"):
            entry["pdf_url"] = entry["link_attachments"][0]

        # Now truly log exactly one line to mismatch.log
        with open("mismatch.log", "a") as logf:
            logf.write(f"{fname} → No reliable metadata found via arXiv/DOI/Title\n")

    # ───── 6) Drop the 8 unwanted keys ─────
    for bad_key in [
        "bibtex_source", "arxiv_id", "docname", "dockey",
        "fields_to_overwrite_from_metadata", "key",
        "bibtex_type", "doc_id"
    ]:
        entry.pop(bad_key, None)

    return make_json_safe(entry)


# ─────────────────────────────────────────────────────────────────────────────
# 5) Async driver
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    updated_count = 0

    for fname, entry in all_meta.items():
        # Skip if already enriched (citation_count not None)
        if entry.get("citation_count") is not None:
            continue

        print(f"⏳ Processing: {fname!r}")
        new_entry = await fill_entry(fname, entry)
        all_meta[fname] = new_entry
        updated_count += 1

    # Write full JSON once, after sanitizing
    safe_meta = make_json_safe(all_meta)
    with open("metadata_trimmed.json", "w", encoding="utf-8") as f:
        json.dump(safe_meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Updated {updated_count} entries → metadata_trimmed.json written")

if __name__ == "__main__":
    asyncio.run(main())
