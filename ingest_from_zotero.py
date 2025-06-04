#!/usr/bin/env python3
"""
ingest_from_zotero.py

1) Read Zotero CSV (zotero.csv), using "Title" as the unique key (slugified).  
2) Pull in exactly the Zotero fields you want (genre, DOI, link attachments, tags, etc.).  
3) Call PaperQA’s DocMetadataClient (or CrossRef fallback) only for missing fields:
     - citation_count, reference_count, is_open_access, publisher, etc.
     - Perform title‐fuzziness and year‐sanity checks; log any mismatches.  
4) Honor manual_overrides.json for any paper you want to hardcode.  
5) Write out metadata_trimmed.json containing only your chosen fields.

Usage:
  python ingest_from_zotero.py
"""

import os
import csv
import json
import asyncio
import requests
from rapidfuzz import fuzz
from dotenv import load_dotenv
from slugify import slugify  # pip install python-slugify
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
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

# ─────────────────────────────────────────────────────────────────────────────
# 1) Fuzzy‐Title → DOI via CrossRef (restricted by first author if given)
# ─────────────────────────────────────────────────────────────────────────────
def find_doi_via_crossref(title_guess: str, first_author: str | None = None) -> str | None:
    """
    Query CrossRef’s /works?query.title=…[&query.author=…] to get the top DOI.
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

# ─────────────────────────────────────────────────────────────────────────────
# 2) Direct CrossRef /works/{doi} lookup (fallback)
# ─────────────────────────────────────────────────────────────────────────────
def crossref_lookup_by_doi(doi: str) -> dict | None:
    """
    Given a DOI, call CrossRef /works/{doi} to get minimal metadata.
    Returns a dict with keys: "title", "authors", "year", "container_title".
    """
    url = f"https://api.crossref.org/works/{doi}"
    params = {}
    if CROSSREF_MAILTO:
        params["mailto"] = CROSSREF_MAILTO

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        item = r.json().get("message", {})
        title = (item.get("title") or [""])[0]
        author_list = []
        for a in item.get("author", []):
            given = a.get("given", "").strip()
            family = a.get("family", "").strip()
            if given or family:
                author_list.append(f"{given} {family}".strip())
        year = item.get("issued", {}).get("date-parts", [[None]])[0][0]
        container = (item.get("container-title") or [""])[0]
        return {
            "title": title,
            "authors": author_list,
            "year": year,
            "container_title": container,
        }
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 3) Load Zotero CSV (zotero.csv) using "Title" as key (slugify to avoid invalid JSON keys)
# ─────────────────────────────────────────────────────────────────────────────
ZOTERO_CSV = "zotero.csv"
all_meta: dict[str, dict] = {}

with open(ZOTERO_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw_title = row.get("Title", "").strip()
        if not raw_title:
            continue

        # Create a slug from the title to use as our dictionary key
        fname = slugify(raw_title, separator="_")

        # Zotero fields we want:
        authors = [a.strip() for a in row.get("Author", "").split(";") if a.strip()]
        doi = row.get("DOI", "").strip() or None
        url = row.get("Url", "").strip() or None
        journal = row.get("Publication Title", "").strip() or None
        year = None
        if row.get("Publication Year", "").isdigit():
            year = int(row["Publication Year"])

        # Item Type → genre
        zotero_type = row.get("Item Type", "").strip()
        genre = zotero_type  # e.g. "journalArticle" or "conferencePaper"

        # Link Attachments (direct PDF URLs)
        link_attachments = [
            l.strip()
            for l in row.get("Link Attachments", "").split(";")
            if l.strip()
        ]

        # Manual and Automatic Tags
        manual_tags = [
            t.strip()
            for t in row.get("Manual Tags", "").split(";")
            if t.strip()
        ]
        automatic_tags = [
            t.strip()
            for t in row.get("Automatic Tags", "").split(";")
            if t.strip()
        ]

        # Pages / Issue / Volume / Publisher / Abstract
        pages = row.get("Pages", "").strip() or None
        issue = row.get("Issue", "").strip() or None
        volume = row.get("Volume", "").strip() or None
        publisher = row.get("Publisher", "").strip() or None
        abstract = row.get("Abstract Note", "").strip() or None

        # Initialize the metadata entry
        all_meta[fname] = {
            "title": raw_title or None,
            "authors": authors or None,
            "doi": doi,
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
            # placeholders for API‐supplied fields:
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
        }

# ─────────────────────────────────────────────────────────────────────────────
# 4) Load manual_overrides.json (if present)
# ─────────────────────────────────────────────────────────────────────────────
MANUAL_OVERRIDES = "manual_overrides.json"
if os.path.exists(MANUAL_OVERRIDES):
    with open(MANUAL_OVERRIDES, "r", encoding="utf-8") as f:
        manual_overrides = json.load(f)
else:
    manual_overrides = {}

# ─────────────────────────────────────────────────────────────────────────────
# 5) Define the async "fill_with_api" routine (no PDF metadata extraction)
# ─────────────────────────────────────────────────────────────────────────────
async def fill_with_api(fname: str, entry: dict) -> dict:
    """
    Fill missing fields in `entry` via PaperQA’s DocMetadataClient → CrossRef fallback.
    Honor manual_overrides first; perform title and year sanity checks.
    Returns the updated entry (possibly unchanged if no API result was good).
    """
    # 5a) If fname is in manual_overrides, merge and return immediately:
    if fname in manual_overrides:
        entry.update(manual_overrides[fname])
        return entry

    client = DocMetadataClient(clients=ALL_CLIENTS)

    # 5b) Build query_kwargs:
    doi = entry.get("doi")
    if doi:
        query_kwargs = {"doi": doi}
    else:
        # No DOI → attempt CrossRef fuzzy‐title (restrict by first author if possible)
        title_guess = entry.get("title") or fname.replace("_", " ")
        authors = entry.get("authors") or []
        first_author = authors[0].split()[-1] if authors else None

        doi_found = find_doi_via_crossref(title_guess, first_author)
        if doi_found:
            query_kwargs = {"doi": doi_found}
            entry["doi"] = doi_found
        else:
            # Fallback: direct match by title + authors
            query_kwargs = {"title": title_guess}
            if authors:
                query_kwargs["authors"] = authors

    # 5c) Rate‐limit: one query at a time + small delay
    details = None
    await asyncio.sleep(1.0)
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
                "genre",
                "bibtex_source",
                "source_quality",
            ],
        )
    except Exception:
        details = None

    # 5d) If we got a result, sanitize + sanity‐check it:
    if details:
        md = details.model_dump()
        md.pop("other", None)  # drop the giant “other” block

        # Title similarity check
        returned_title = (md.get("title") or "").lower()
        guess = (entry.get("title") or "").lower()
        if not guess:
            guess = fname.replace("_", " ").lower()
        sim = fuzz.partial_ratio(guess, returned_title)
        if sim < 60:
            with open("mismatch.log", "a", encoding="utf-8") as log:
                log.write(
                    f"{fname} → Title mismatch: guessed “{guess}” vs returned “{returned_title}” (score {sim})\n"
                )
        else:
            # Year sanity check
            guessed_year = entry.get("year")
            returned_year = md.get("year")
            if guessed_year and returned_year and abs(returned_year - guessed_year) > 1:
                with open("mismatch.log", "a", encoding="utf-8") as log:
                    log.write(
                        f"{fname} → Year mismatch: guessed {guessed_year} vs returned {returned_year}\n"
                    )
            else:
                # Merge all non‐None returned fields into entry
                for k, v in md.items():
                    if v is not None:
                        entry[k] = v
                return entry

    # 5e) If API gave nothing useful, fallback to direct CrossRef /works/{doi}, if we now have a DOI
    doi = entry.get("doi")
    if doi:
        cr = crossref_lookup_by_doi(doi)
        if cr:
            entry["title"] = cr["title"]
            entry["authors"] = cr["authors"]
            entry["year"] = cr["year"]
            if not entry.get("journal"):
                entry["journal"] = cr.get("container_title")
            return entry

    # 5f) Still nothing → log it and move on
    with open("mismatch.log", "a", encoding="utf-8") as log:
        log.write(f"{fname} → No reliable metadata found via API.\n")
    return entry

# ─────────────────────────────────────────────────────────────────────────────
# 6) The async driver: loop through all_meta keys and fill missing fields
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    updated_count = 0
    for fname, entry in all_meta.items():
        # Only attempt API if any of these core fields are still None
        if (
            entry.get("citation_count") is None
            or entry.get("reference_count") is None
            or entry.get("is_open_access") is None
            or not entry.get("publisher")
        ):
            updated = await fill_with_api(fname, entry)
            all_meta[fname] = updated
            updated_count += 1

        # 6a) If no pdf_url from API, but Zotero’s link_attachments has something, use that
        if not entry.get("pdf_url") and entry.get("link_attachments"):
            entry["pdf_url"] = entry["link_attachments"][0]

    # 7) Write out the final, pruned JSON
    with open("metadata_trimmed.json", "w", encoding="utf-8") as f:
        json.dump(all_meta, f, indent=2)

    print(f"✅ Updated {updated_count} entries; wrote metadata_trimmed.json")

if __name__ == "__main__":
    asyncio.run(main())
