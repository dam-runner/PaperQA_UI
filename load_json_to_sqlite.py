import json
import sqlite3
import os
from datetime import date

# 1. Load the JSON file
json_path = "metadata_trimmed.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    # Handle two possible structures:
    # (a) a list of paper‐objects
    # (b) a dict of {paperKey: paper‐object}
    if isinstance(data, dict):
        papers_list = list(data.values())
    else:
        papers_list = data

# 2. Connect to (or create) SQLite database
db_path = "asr_papers.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 3. Create the 'papers' table
cursor.execute("""
CREATE TABLE IF NOT EXISTS papers (
    paper_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    title               TEXT NOT NULL,
    genre               TEXT,
    authors             TEXT,
    year                INTEGER,
    url                 TEXT,
    publisher           TEXT,
    journal             TEXT,
    doi                 TEXT UNIQUE,
    abstract            TEXT,
    doi_url             TEXT,
    formatted_citation  TEXT,
    citation_count      INTEGER,
    source_quality      TEXT,
    link_attachments    TEXT,
    date_added          DATE DEFAULT (DATE('now'))
);
""")
conn.commit()

# 4. Insert each paper into SQLite
for rec in papers_list:
    title = rec.get("title") or "Unknown"
    genre = rec.get("genre") or "Unknown"

    # authors: if it's a list, join with semicolons; else treat as string
    authors_val = rec.get("authors")
    if isinstance(authors_val, list):
        authors_str = "; ".join(authors_val)
    elif isinstance(authors_val, str):
        authors_str = authors_val
    else:
        authors_str = "Unknown"

    # year: try to parse as int; if not parseable or missing, use NULL
    try:
        year_val = int(rec.get("year"))
    except (TypeError, ValueError):
        year_val = None

    url = rec.get("url") or None
    publisher = rec.get("publisher") or None
    journal = rec.get("journal") or None
    doi = rec.get("doi") or None
    abstract = rec.get("abstract") or None
    doi_url = rec.get("doi_url") or None
    formatted_citation = rec.get("formatted_citation") or None

    # citation_count: parse int or None
    try:
        citation_count = int(rec.get("citation_count"))
    except (TypeError, ValueError):
        citation_count = None

    source_quality = rec.get("source_quality") or None

    # link_attachments: if it’s a list, store as JSON string; else if string, store as single‐element list
    la = rec.get("link_attachments")
    if isinstance(la, list):
        link_attachments_str = json.dumps(la)  # store JSON array text
    elif isinstance(la, str):
        link_attachments_str = json.dumps([la])
    else:
        link_attachments_str = None

    # 5. Insert or ignore duplicates by DOI
    cursor.execute("""
    INSERT OR IGNORE INTO papers
    (title, genre, authors, year, url, publisher, journal, doi, abstract, doi_url,
     formatted_citation, citation_count, source_quality, link_attachments)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        title, genre, authors_str, year_val, url, publisher, journal, doi,
        abstract, doi_url, formatted_citation, citation_count, source_quality,
        link_attachments_str
    ))

conn.commit()
conn.close()

print(f"Loaded {len(papers_list)} records into {db_path}.")
