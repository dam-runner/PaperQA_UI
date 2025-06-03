#!/usr/bin/env python3
"""
extract_full_metadata.py

For each PDF in PAPERS_PATH, fetch “full” metadata by:

  1. Looking up the DOI via CrossRef (fuzzy-search on title).
  2. If a DOI is found, call DocMetadataClient.query(doi=…).
  3. If no DOI is found, fall back to DocMetadataClient.query(title=…, authors=…).

Merges into metadata.json (if present), skipping any entries that already have a “doi”.

Any Python `set` or other non‐JSON‐primitive fields are converted to JSON‐safe types
before writing metadata.json.

Usage:
  1. Ensure .env contains:
       OPENAI_API_KEY=sk-...
       PAPERS_PATH="/absolute/path/to/your/pdf/folder"
       CROSSREF_MAILTO="your_email@example.com"
  2. Install deps if needed:
       pip install paperqa aiohttp python-dotenv requests
  3. Run:
       python extract_full_metadata.py

To force a full refresh, delete or rename metadata.json before running.
"""

import os
import json
import glob
import asyncio
import aiohttp
import requests
import time
from dotenv import load_dotenv
from paperqa.clients import DocMetadataClient, ALL_CLIENTS

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load environment variables
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PAPERS_PATH = os.getenv("PAPERS_PATH")
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", None)

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")
if not PAPERS_PATH or not os.path.isdir(PAPERS_PATH):
    raise RuntimeError(f"Invalid PAPERS_PATH in .env: {PAPERS_PATH}")

PAPERS_PATH = os.path.expanduser(PAPERS_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Load existing metadata.json (if present), back up if invalid JSON
# ─────────────────────────────────────────────────────────────────────────────
META_FILE = "metadata.json"
all_meta = {}
if os.path.exists(META_FILE):
    try:
        with open(META_FILE, "r", encoding="utf-8") as f:
            all_meta = json.load(f)
    except json.JSONDecodeError:
        # Backup the broken file, start fresh
        backup_name = f"metadata_broken_{int(time.time())}.json"
        os.rename(META_FILE, backup_name)
        print(f"⚠️  {META_FILE} was invalid JSON. Renamed to {backup_name}. Starting fresh.")
        all_meta = {}

# ─────────────────────────────────────────────────────────────────────────────
# 3) Build “todos”: PDFs that lack a DOI in metadata.json
# ─────────────────────────────────────────────────────────────────────────────
pdf_paths = glob.glob(os.path.join(PAPERS_PATH, "*.pdf"))
todos = []

for full_path in pdf_paths:
    fname = os.path.basename(full_path)  # e.g. "Smith_2021_Multilingual_ASR.pdf"
    existing = all_meta.get(fname, {})

    if existing.get("doi"):
        # Already has a DOI → assume full metadata present
        continue

    # If metadata.json already has an arxiv_id, use that
    if existing.get("arxiv_id"):
        todos.append({"key": fname, "arxiv_id": existing["arxiv_id"]})
        continue

    # If metadata.json already has a “title” and “authors”, use that as fallback
    title = existing.get("title", "").strip()
    authors = existing.get("authors", [])
    if title:
        q = {"key": fname, "title": title}
        if isinstance(authors, list) and authors:
            q["authors"] = authors
        todos.append(q)
    else:
        # As a last resort, guess from filename
        base = os.path.splitext(fname)[0]
        guess = base.replace("_", " ").replace("-", " ")
        todos.append({"key": fname, "title": guess})

if not todos:
    print("✅ All PDFs already have a DOI → nothing to update.")
    print("If you want to re-run for every paper, delete/rename metadata.json.")
    exit(0)

print(f"→ Found {len(todos)} PDFs requiring metadata lookup (out of {len(pdf_paths)}).")

# ─────────────────────────────────────────────────────────────────────────────
# 4) Define a CrossRef helper to find DOI by fuzzy‐title search
# ─────────────────────────────────────────────────────────────────────────────
def find_doi_via_crossref(title_guess: str) -> str | None:
    """
    Given a title guess, query CrossRef (/works?query.title=…) for the top hit,
    return its DOI string if found, else None.
    """
    params = {"query.title": title_guess, "rows": 1}
    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO

    try:
        r = requests.get("https://api.crossref.org/works", params=params, timeout=10)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        if items:
            doi = items[0].get("DOI")
            return doi
    except Exception as e:
        print(f"⚠️ CrossRef lookup failed for title '{title_guess}': {e}")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 5) Define rate-limited fetch: DOI‐first, then DocMetadataClient
# ─────────────────────────────────────────────────────────────────────────────
sem = asyncio.Semaphore(1)

async def fetch_meta(client: DocMetadataClient, item: dict):
    """
    1) If item has “doi” already, just call client.query(doi=…).
    2) Else if item has “title” (and possibly “authors”), first try CrossRef to get a DOI.
       - If CrossRef returns a DOI, call client.query(doi=…).
       - If not, call client.query(title=…, authors=…) as fallback.
    Returns (key, sanitized_metadata_dict) or (key, None).
    """
    key = item["key"]
    # Step 1: Attempt to get DOI from existing item or CrossRef
    doi = item.get("doi")
    if not doi:
        # If we already have a “title” or “authors” from metadata.json, use that
        title = item.get("title", "")
        doi = find_doi_via_crossref(title)
        if doi:
            # Record the discovered DOI in existing entry so we skip next time
            print(f"✅ CrossRef found DOI for '{key}': {doi}")
        else:
            print(f"ℹ️  CrossRef did NOT find a DOI for '{key}' (using title fallback).")

    # Next, build query_kwargs for DocMetadataClient
    if doi:
        query_kwargs = {"doi": doi}
    elif "arxiv_id" in item:
        query_kwargs = {"arxiv_id": item["arxiv_id"]}
    else:
        # Fallback: title + optional authors
        query_kwargs = {"title": item.get("title", "")}
        if item.get("authors"):
            query_kwargs["authors"] = item["authors"]

    # Rate-limit: one at a time + 3s delay
    async with sem:
        await asyncio.sleep(3.0)
        try:
            details = await client.query(
                **query_kwargs,
                fields=[
                    "title",
                    "authors",
                    "year",
                    "doi",
                    "venue",
                    "citation_count",
                    "reference_count",
                    "is_open_access",
                    "pdf_url",
                    "license",
                    "fields_of_study",
                    "crossref_subjects",
                    "publisher",
                    "journal_volume",
                    "journal_issue",
                    "journal_pages",
                    "arxiv_primary_category",
                    "journal_ref",
                    "num_pages",
                    "topics",
                ],
            )
        except Exception as e:
            print(f"❌ DocMetadataClient error for '{key}' ({query_kwargs}): {e}")
            return key, None

        if not details:
            print(f"⚠️  No metadata found for '{key}' ({query_kwargs})")
            return key, None

        md = details.model_dump()

        # ─────────────────────────────────────────────────────────────────────────
        # 5a) Remove purely internal fields
        # ─────────────────────────────────────────────────────────────────────────
        for fld in [
            "fields_to_overwrite_from_metadata",
            "dockey",
            "docname",
            "citation",
        ]:
            md.pop(fld, None)

        # ─────────────────────────────────────────────────────────────────────────
        # 5b) Recursively sanitize (set/tuple → list, others → str if needed)
        # ─────────────────────────────────────────────────────────────────────────
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, (list,)):
                return [sanitize(v) for v in obj]
            elif isinstance(obj, (set, tuple)):
                return [sanitize(v) for v in obj]
            elif obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            else:
                # Fallback: convert to string
                return str(obj)

        md = sanitize(md)

        return key, md

# ─────────────────────────────────────────────────────────────────────────────
# 6) Async driver: loop through todos, merge into all_meta
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    client = DocMetadataClient(clients=ALL_CLIENTS)
    updated_count = 0

    for item in todos:
        key = item["key"]
        print(f"Fetching metadata for '{key}'…")
        k, data = await fetch_meta(client, item)

        if data:
            # Merge into existing entry (if any)
            merged = all_meta.get(k, {}).copy()
            merged.update(data)
            all_meta[k] = merged
            updated_count += 1

    # 7) Write out metadata.json
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\n✅ Done: Added/updated {updated_count} entries.")
    print(f"Final metadata.json contains {len(all_meta)} entries.")

if __name__ == "__main__":
    asyncio.run(main())
