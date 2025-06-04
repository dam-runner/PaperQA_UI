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
    1) If manual override exists for fname, merge and return.
    2) Step 0: If Zotero URL is arXiv, query by arXiv ID.
    3) Step 1: If Zotero DOI present, query by that DOI exactly once.
    4) Step 2: Clean title, ask CrossRef for a new DOI, then query by that DOI.
    5) Step 3: Last‐resort title+authors query with high fuzzy threshold.
    6) Step 4: Fallback to Zotero-only. Log every failure to mismatch.log.
    """
    # 4a) Apply manual override if it exists
    if fname in manual_overrides:
        entry.update(manual_overrides[fname])
        return make_json_safe(entry)

    client = DocMetadataClient(clients=ALL_CLIENTS)

    # STEP 0: If Zotero URL is an arXiv link, do arXiv lookup first
    url = entry.get("url", "")
    arxiv_id = extract_arxiv_id(url)
    if arxiv_id:
        entry["arxiv_id"] = arxiv_id
        print(f">>> [{fname}] Identified Zotero URL as arXiv: {arxiv_id}")
        async with sem:
            await asyncio.sleep(1.0)
            try:
                details = await client.query(arxiv_id=arxiv_id, fields=[
                    "title", "authors", "year", "doi", "citation_count",
                    "reference_count", "is_open_access", "pdf_url",
                    "fields_of_study", "crossref_subjects", "publisher",
                    "container_title", "bibtex_source", "source_quality",
                ])
            except Exception as e:
                print(f"    ❌ arXiv query error for {arxiv_id}: {e}")
                details = None

        if details:
            md = details.model_dump()
            md.pop("other", None)
            for k, v in md.items():
                if v is not None:
                    entry[k] = v
            return make_json_safe(entry)
        else:
            with open("mismatch.log", "a", encoding="utf-8") as log:
                log.write(f"{fname} → arXiv ID {arxiv_id} returned no data\n")

    # STEP 1: If Zotero DOI is present, try exactly that once
    zotero_doi = entry.get("doi")
    if zotero_doi:
        print(f">>> [{fname}] Trying Zotero DOI: {zotero_doi}")
        async with sem:
            await asyncio.sleep(1.0)
            try:
                details = await client.query(doi=zotero_doi, fields=[
                    "title", "authors", "year", "doi", "citation_count",
                    "reference_count", "is_open_access", "pdf_url",
                    "fields_of_study", "crossref_subjects", "publisher",
                    "container_title", "bibtex_source", "source_quality",
                ])
            except Exception as e:
                print(f"    ❌ Error querying Zotero DOI {zotero_doi}: {e}")
                details = None

        if details:
            md = details.model_dump()
            md.pop("other", None)
            for k, v in md.items():
                if v is not None:
                    entry[k] = v
            return make_json_safe(entry)
        else:
            with open("mismatch.log", "a", encoding="utf-8") as log:
                log.write(f"{fname} → Zotero DOI {zotero_doi} returned no data\n")

    # STEP 2: CrossRef “clean title → new DOI → query”
    raw = entry.get("title", "")
    cleaned = clean_title(raw)
    first_author = (entry.get("authors") or [None])[0]
    if first_author:
        first_author = first_author.split()[-1]  # last name only

    print(f">>> [{fname}] CrossRef lookup for “{cleaned}” …")
    found_doi = find_doi_via_crossref(cleaned, first_author)
    if found_doi:
        print(f"    ✅ CrossRef returned DOI = {found_doi}")
        entry["doi"] = found_doi
        async with sem:
            await asyncio.sleep(1.0)
            try:
                details = await client.query(doi=found_doi, fields=[
                    "title", "authors", "year", "doi", "citation_count",
                    "reference_count", "is_open_access", "pdf_url",
                    "fields_of_study", "crossref_subjects", "publisher",
                    "container_title", "bibtex_source", "source_quality",
                ])
            except Exception as e:
                print(f"    ❌ Error querying CrossRef DOI {found_doi}: {e}")
                details = None

        if details:
            md = details.model_dump()
            md.pop("other", None)
            for k, v in md.items():
                if v is not None:
                    entry[k] = v
            return make_json_safe(entry)
        else:
            with open("mismatch.log", "a", encoding="utf-8") as log:
                log.write(f"{fname} → CrossRef DOI {found_doi} returned no data\n")
    else:
        with open("mismatch.log", "a", encoding="utf-8") as log:
            log.write(f"{fname} → CrossRef lookup for “{cleaned}” found no DOI\n")

    # STEP 3: Last‐resort title+author query (only if no DOI found)
    if not entry.get("doi"):
        threshold = 85
        print(f">>> [{fname}] Last‐resort title+author query on “{cleaned}”")
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
            except Exception as e:
                print(f"    ❌ Error in title+author query: {e}")
                details = None

        if details:
            md = details.model_dump()
            ret_title = (md.get("title") or "").lower()
            sim = fuzz.partial_ratio(cleaned.lower(), ret_title)
            if sim >= threshold:
                md.pop("other", None)
                for k, v in md.items():
                    if v is not None:
                        entry[k] = v
                return make_json_safe(entry)
            else:
                with open("mismatch.log", "a", encoding="utf-8") as log:
                    log.write(f"{fname} → title+author fuzzy match too low ({sim} < {threshold})\n")
        else:
            with open("mismatch.log", "a", encoding="utf-8") as log:
                log.write(f"{fname} → title+author query returned no data\n")

    # STEP 4: Give up and return Zotero‐only fields
    with open("mismatch.log", "a", encoding="utf-8") as log:
        log.write(f"{fname} → No reliable metadata found via arXiv/DOI/Title\n")

    # If no pdf_url from API but Zotero has link attachments, use the first
    if not entry.get("pdf_url") and entry.get("link_attachments"):
        entry["pdf_url"] = entry["link_attachments"][0]

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
