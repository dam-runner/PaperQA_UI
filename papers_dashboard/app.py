# === In papers_dashboard/app.py ===

import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import sqlite3
import json
import matplotlib.pyplot as plt
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# ——————————————————————————————————————————
# 1) Database connection & global variables
# ——————————————————————————————————————————
DB_PATH = "asr_papers.db"

# Config Settings
st.set_page_config(
    page_title="Child/Multilingual ASR Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Troubleshooting
#st.sidebar.write("Interpreter:", sys.executable)

@st.cache_resource
def get_connection():
    """Return a cached connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

conn = get_connection()

# Compute total_papers once, so every page can reference it
total_papers = int(pd.read_sql("SELECT COUNT(*) AS c FROM papers;", conn)["c"].iloc[0])


# ——————————————————————————————————————————
# 2) Normalization helpers
# ——————————————————————————————————————————
def normalize_genre(raw_genre):
    """Map raw genre codes to a human-readable form."""
    if not raw_genre or raw_genre.strip() == "":
        return "Unknown"
    # NOTE: If you prefer to load this mapping from a .txt file, write a small loader
    # function and store the raw->normalized pairs in that file.
    mapping = {
        "book":            "Book",
        "bookSection":     "Book Section",
        "conferencePaper": "Conference Paper",
        "journalArticle":  "Journal Article",
        "preprint":        "Preprint",
        "webpage":         "Webpage",
    }
    # Return the mapped value if present, otherwise return raw_genre unchanged
    return mapping.get(raw_genre, raw_genre)


def normalize_journal(raw_journal):
    """Map raw journal strings (often long conference names) to a short label."""
    if not raw_journal or raw_journal.strip() == "":
        return "Unknown"
    key = raw_journal.strip().lower()

    # Substring-based grouping rules. Order matters: more specific first.
    if "icassp" in key:
        return "ICASSP"
    if "interspeech" in key:
        return "Interspeech"
    if "eurospeech" in key:
        return "Eurospeech"
    if "icslp" in key:
        return "ICSLP"
    if "sltu" in key:
        return "SLTU"
    if "arxiv" in key:
        return "ArXiv"
    if "acm transactions on accessible computing" in key:
        return "ACM TO Access Comp"
    if "american journal of speech-language pathology" in key:
        return "Am. J. Speech-Lang. Path."
    if "applied sciences" in key:
        return "Applied Sciences"
    if "behavior research methods" in key:
        return "Behavior Research Methods"
    if "blackwell handbook of language development" in key:
        return "Blackwell Handbook of Lang Dev"
    if "child development" in key:
        return "Child Development"
    if "clinical sociolinguistics" in key:
        return "Clinical Sociolinguistics"
    if "comput. speech" in key or "computer speech & language" in key:
        return "Comput. Speech & Lang"
    if "computational linguistics" in key:
        return "Computational Linguistics"
    if "ecs transactions" in key:
        return "ECS Transactions"
    if "eurasip" in key:
        return "EURASIP J. Audio Speech Music Process"
    if "folia phoniatrica et logopaedica" in key:
        return "Folia Phoniatrica et Logopaedica"
    if "ieee access" in key:
        return "IEEE Access"
    if "journal of selected topics in signal processing" in key:
        return "IEEE JSTSP"
    if "speech and audio processing" in key or "transactions on speech" in key:
        return "IEEE TASAP"
    if "acoustical society of america" in key:
        return "JASA"
    if "int’l journal of bilingual education" in key or "bilingualism" in key:
        return "Int’l J. Bilingual Ed & Bilingualism"
    if "international journal of multilingualism" in key:
        return "International Journal of Multilingualism"
    if "journal of child language" in key:
        return "Journal of Child Language"
    if "journal of childhood and adolescence research" in key:
        return "Journal of Childhood & Adolescence Research"
    if "jslhr" in key or "journal of speech, language, and hearing research" in key:
        return "JSLHR"
    if "multilingual aspects of speech sound disorders" in key:
        return "Multilingual Aspects of Speech Sound Disorders in Children"
    if "multimedia tools and applications" in key:
        return "Multimedia Tools & Applications"
    if "natural language engineering" in key:
        return "Natural Language Engineering"
    if "natural science" in key:
        return "Natural Science"
    if "open mind" in key:
        return "Open Mind: Discoveries in Cognitive Science"
    if "icon" in key:
        return "ICON Proceedings"
    if "sigkdd" in key or "knowledge discovery" in key:
        return "KDD Proceedings"
    if "child, computer and interaction" in key:
        return "Child-Computer Interaction Proceedings"
    if "machine learning" in key:
        return "ICML Proceedings"
    if "ai ethics" in key:
        return "AAAI/ACM AI Ethics Proceedings"
    if "lrec" in key:
        return "LREC Proceedings"
    if "reimagining innovation" in key:
        return "Reimagining Innovation in Ed & Social Sci"
    if "researchgate" in key:
        return "ResearchGate"
    if "social policy report" in key:
        return "Social Policy Report"
    if "speech commun" in key:
        return "Speech Communication"
    if "handbook of bilingualism and multilingualism" in key:
        return "Handbook of Bilingualism & Multilingualism"
    if "handbook of child language" in key:
        return "Handbook of Child Language"

    # If no rule matched, just show the raw string
    return raw_journal


# (We’ll omit normalize_source_quality entirely since we choose not to surface that.)
# (normalize_publisher and safe_str are unused in this Stage 1 MVP; remove them if you like.)



# ——————————————————————————————————————————
# 3) Sidebar navigation
# ——————————————————————————————————————————
st.sidebar.title("Navigation")
menu = ["Overview", "Charts", "Paper List"]
choice = st.sidebar.radio("Go to", menu)



# ========================================================
# 4) Overview Page
# ========================================================
if choice == "Overview":
    st.title("Overview: Child/Multilingual ASR Papers (2015–2025)")

    # 4.1) Papers per Year
    df_year = pd.read_sql("SELECT year FROM papers;", conn)
    missing_year_count = df_year["year"].isna().sum()
    df_year_valid = df_year.dropna(subset=["year"])
    year_counts = df_year_valid["year"].astype(int).value_counts().sort_index()

    st.subheader("Number of Papers per Year")
    st.bar_chart(year_counts)
    st.caption(f"⚠️ {missing_year_count} of {total_papers} papers have an unknown year.")

    # 4.2) Citation Count Distribution
    bins = [0, 10, 50, 100, 500, 1000, float("inf")]
    labels = ["1–10", "11–50", "51–100", "101–500", "501–1000", ">1000"]
    df_cite = pd.read_sql("SELECT citation_count FROM papers WHERE citation_count IS NOT NULL;", conn)
    df_cite["bucket"] = pd.cut(df_cite["citation_count"], bins=bins, labels=labels, right=True)
    counts = df_cite["bucket"].value_counts().reindex(labels, fill_value=0)
    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Citation Count Range")
    ax.set_ylabel("Number of Papers")
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    st.pyplot(fig)
    st.caption(f"⚠️ {df_cite['citation_count'].isna().sum()} papers missing citation_count.")


    # 4.3) Papers per Author (Top 10)

    df_auth = pd.read_sql("SELECT authors FROM papers;", conn)

    def clean_author(a):
        if not a or not str(a).strip():
            return None
        s = str(a).strip()
        sl = s.lower()
        if "none" in sl or "unknown" in sl:
            return None
        return s

    # apply cleaning
    df_auth["authors_clean"] = df_auth["authors"].apply(clean_author)

    # count how many got dropped
    missing = df_auth["authors_clean"].isna().sum()
    total   = len(df_auth)

    # build a flat list of all author names (one entry per paper-author pair)
    all_authors = []
    for auth in df_auth["authors_clean"].dropna():
        for name in auth.split(";"):
            name = name.strip()
            if name:
                all_authors.append(name)

    # count and take top 10
    author_counts = pd.Series(all_authors).value_counts().head(10)

    # plot
    st.subheader("Number of Papers per Author (Top 10)")
    fig, ax = plt.subplots()
    ax.barh(author_counts.index, author_counts.values)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Papers")
    ax.set_ylabel("Author")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    st.pyplot(fig)

    st.caption(f"Excluded {missing} of {total} papers with unknown authors.")

    # 4.4) Citation counts per author

    df_auth = pd.read_sql("SELECT authors, citation_count FROM papers;", conn)

    # ensure citation_count is integer
    df_auth["citation_count"] = (
        df_auth["citation_count"]
        .fillna(0)
        .astype(int)
    )

    # safe author-cleaning function
    def clean_author(a):
        # treat None, empty, whitespace-only as missing
        if not a or not str(a).strip():
            return None
        s = str(a).strip()
        s_lower = s.lower()
        # drop any that mention “none” or “unknown”
        if "none" in s_lower or "unknown" in s_lower:
            return None
        return s

    # apply cleaning
    df_auth["authors_clean"] = df_auth["authors"].apply(clean_author)

    # count how many rows became missing
    missing_authors = df_auth["authors_clean"].isna().sum()
    total_papers    = df_auth.shape[0]

    # build a list of (author, citations) pairs
    author_cits = []
    for _, row in df_auth.dropna(subset=["authors_clean"]).iterrows():
        for name in row["authors_clean"].split(";"):
            name = name.strip()
            if name:
                author_cits.append((name, row["citation_count"]))

    # turn into DataFrame and sum citations per author
    df_ac = pd.DataFrame(author_cits, columns=["author","citations"])
    top_authors = (
        df_ac
        .groupby("author")["citations"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    # plot
    st.subheader("Top 10 Authors by Total Citation Count")
    fig, ax = plt.subplots()
    ax.barh(top_authors.index, top_authors.values)
    ax.invert_yaxis()
    ax.set_xlabel("Total Citations")
    ax.set_ylabel("Author")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    st.pyplot(fig)

    st.caption(f"Excludes {missing_authors} out of {total_papers} papers with unknown authors.")


    # 4.5) Papers by Genre
    df_genre_raw = pd.read_sql("SELECT genre FROM papers;", conn)
    normalized_genres = df_genre_raw["genre"].apply(lambda g: normalize_genre(g))
    genre_counts = normalized_genres.value_counts()
    missing_genre = (df_genre_raw["genre"].isna() | (df_genre_raw["genre"].str.strip() == "")).sum()

    st.subheader("Papers by Genre")
    fig_genre, ax_genre = plt.subplots()
    ax_genre.pie(
        genre_counts.values,
        labels=genre_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax_genre.axis("equal")
    st.pyplot(fig_genre)
    st.caption(f"⚠️ {missing_genre} of {total_papers} papers have unknown genre.")

    # 4.6) Papers by Journal (Top 10)
    df_j = pd.read_sql("SELECT journal FROM papers;", conn)
    norm_j = df_j["journal"].apply(normalize_journal)
    missing_j = (df_j["journal"].isna() | (df_j["journal"].str.strip() == "")).sum()
    counts = norm_j[norm_j!="Unknown"].value_counts().head(10)
    fig, ax = plt.subplots()
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)
    st.caption(f"⚠️ {missing_j} papers have unknown journal.")


# ========================================================
# 5) Charts Page (All Charts Together)
# ========================================================
elif choice == "Charts":
    st.title("All Charts & Distributions")

    # (1) Papers per Year
    st.subheader("Papers per Year")
    df_year = pd.read_sql("SELECT year FROM papers;", conn)
    missing_year = df_year["year"].isna().sum()
    valid_year = df_year.dropna(subset=["year"])["year"].astype(int)
    counts_year = valid_year.value_counts().sort_index()
    st.bar_chart(counts_year)
    st.caption(f"⚠️ {missing_year} of {total_papers} papers have an unknown year.")

    # (2) Citation Count
    st.subheader("Citation Count Distribution")

    # Load the raw citation_count column
    df_cite = pd.read_sql("SELECT citation_count FROM papers;", conn)
    
    # Compute how many are missing
    missing_cite = df_cite["citation_count"].isna().sum()

    # Cast to int for the valid ones
    valid_cite = df_cite["citation_count"].dropna().astype(int)

    # Binned citation counts:
    bins = [0,10,50,100,500,1000,float("inf")]
    labels = ["1–10","11–50","51–100","101–500","501–1000",">1000"]
    df_cite["bucket"] = pd.cut(valid_cite, bins=bins, labels=labels, right=True)
    counts_cite = df_cite["bucket"].value_counts().reindex(labels, fill_value=0)
    fig_cite2, ax_cite2 = plt.subplots()
    ax_cite2.bar(counts_cite.index.astype(str), counts_cite.values)
    ax_cite2.set_xlabel("Citation Count")
    ax_cite2.set_ylabel("# of Papers")
    ax_cite2.set_xticks(range(len(counts_cite)))
    ax_cite2.set_xticklabels(counts_cite.index.astype(str), rotation=45, ha="right")
    st.pyplot(fig_cite2)
    st.caption(f"⚠️ {missing_cite} of {total_papers} papers have unknown citation counts.")

    # (3) Papers per Author (Top 10)
    st.subheader("Papers per Author (Top 10)")

    df_auth = pd.read_sql("SELECT authors FROM papers;", conn)

    # Normalize any “none none” or blank to None
    df_auth["authors_clean"] = df_auth["authors"].apply(
        lambda a: a if isinstance(a, str) and a.strip() and a.lower() != "none none" else None
    )

    # Count missing
    missing_authors = df_auth["authors_clean"].isna().sum()

    # Flatten into one list of author names
    all_authors = []
    for a in df_auth["authors_clean"].dropna():
        all_authors += [name.strip() for name in a.split(";") if name.strip()]

    # Compute frequencies and take top 10
    author_counts = pd.Series(all_authors).value_counts()
    top_authors = author_counts.head(10)

    st.subheader("Number of Papers per Author (Top 10)")
    fig_auth, ax_auth = plt.subplots()
    ax_auth.barh(top_authors.index, top_authors.values)
    ax_auth.invert_yaxis()
    ax_auth.set_xlabel("Number of Papers")
    ax_auth.set_ylabel("Author")
    st.pyplot(fig_auth)

    st.caption(f"Top 10 authors out of {total_papers} papers.")
    st.caption(f"⚠️ {missing_authors} papers have unknown authors.")


    # (4) Papers by Genre
    st.subheader("Papers by Genre")
    df_genre_raw = pd.read_sql("SELECT genre FROM papers;", conn)
    normalized_genres = df_genre_raw["genre"].apply(lambda g: normalize_genre(g))
    genre_counts = normalized_genres.value_counts()
    missing_gen = (df_genre_raw["genre"].isna() | (df_genre_raw["genre"].str.strip() == "")).sum()
    fig_gen2, ax_gen2 = plt.subplots()
    ax_gen2.pie(genre_counts.values, labels=genre_counts.index, autopct="%1.1f%%", startangle=90)
    ax_gen2.axis("equal")
    st.pyplot(fig_gen2)
    st.caption(f"⚠️ {missing_gen} of {total_papers} papers have unknown genre.")

    # (5) Papers by Journal (Top 10)
    st.subheader("Papers by Journal (Top 10)")
    df_jr_raw = pd.read_sql("SELECT journal FROM papers;", conn)
    normalized_journal = df_jr_raw["journal"].apply(lambda j: normalize_journal(j))
    journal_counts = normalized_journal.value_counts().head(10)
    missing_jr = (df_jr_raw["journal"].isna() | (df_jr_raw["journal"].str.strip() == "")).sum()
    fig_jr2, ax_jr2 = plt.subplots()
    ax_jr2.pie(journal_counts.values, labels=journal_counts.index, autopct="%1.1f%%", startangle=90)
    ax_jr2.axis("equal")
    st.pyplot(fig_jr2)
    st.caption(f"⚠️ {missing_jr} of {total_papers} papers have unknown or rare journals.")



# ========================================================
# 6) Paper List (Filter & Search) Page
# ========================================================
elif choice == "Paper List":
    st.title("Paper List (Filter & Search)")

    # ─── Sidebar Filters ──────────────────────────────────────────
    st.sidebar.header("Filter Papers")
    search_kw = st.sidebar.text_input("Search abstracts for keyword").strip()

    year_min = int(pd.read_sql("SELECT MIN(year) AS min_y FROM papers;", conn)["min_y"].iloc[0] or 2015)
    year_max = int(pd.read_sql("SELECT MAX(year) AS max_y FROM papers;", conn)["max_y"].iloc[0] or 2025)
    selected_years = st.sidebar.slider(
        "Publication Year Range",
        min_value=year_min, max_value=year_max,
        value=(year_min, year_max)
    )

    raw_genres   = pd.read_sql("SELECT DISTINCT genre FROM papers;", conn)["genre"].fillna("").tolist()
    norm_genres  = sorted({normalize_genre(g) for g in raw_genres})
    sel_genres   = st.sidebar.multiselect("Genre", options=norm_genres, default=norm_genres)

    raw_journals = pd.read_sql("SELECT DISTINCT journal FROM papers;", conn)["journal"].fillna("").tolist()
    norm_journals= sorted({normalize_journal(j) for j in raw_journals})
    sel_journals = st.sidebar.multiselect("Journal", options=norm_journals, default=norm_journals)

    raw_pubs     = pd.read_sql("SELECT DISTINCT publisher FROM papers;", conn)["publisher"].fillna("").tolist()
    norm_pubs    = sorted(set(raw_pubs))
    sel_pubs     = st.sidebar.multiselect("Publisher", options=norm_pubs, default=norm_pubs)

    # ─── Load & Filter Data ─────────────────────────────────────
    df_all = pd.read_sql(
        "SELECT * FROM papers WHERE year BETWEEN ? AND ?", conn, params=selected_years
    )
    df_all["citation_count"] = df_all["citation_count"].fillna(0).astype(int)
    df_all["genre_norm"]   = df_all["genre"].apply(normalize_genre)
    df_all["journal_norm"] = df_all["journal"].apply(normalize_journal)

    if set(sel_genres) != set(norm_genres):
        df_all = df_all[df_all["genre_norm"].isin(sel_genres)]
    if set(sel_journals) != set(norm_journals):
        df_all = df_all[df_all["journal_norm"].isin(sel_journals)]
    if set(sel_pubs) != set(norm_pubs):
        df_all = df_all[df_all["publisher"].isin(sel_pubs)]
    if search_kw:
        df_all = df_all[df_all["abstract"]
                        .fillna("")
                        .str.lower()
                        .str.contains(search_kw.lower())]

    # ─── Prepare Display DataFrame ─────────────────────────────
    rename_map = {
        "title":             "Title",
        "genre_norm":        "Genre",
        "authors":           "Authors",
        "year":              "Year",
        "publisher":         "Publisher",
        "journal_norm":      "Journal",
        "formatted_citation":"Citation",
        "citation_count":    "Citations",
        "url":               "Link",
        "link_attachments":  "PDF",
        "abstract":          "Abstract",
    }
    cols = list(rename_map.keys())
    df = (
      df_all[cols]
      .rename(columns=rename_map)
      .fillna("")
    )

    # — clickable 🔗 and PDF anchors
    df["Link"] = df["Link"].apply(
      lambda u: f'<a href="{u}" target="_blank">🔗</a>' if u.startswith("http") else ""
    )
    def make_pdfs(cell):
      try:
        arr = json.loads(cell)
        return " ".join(f'<a href="{u}" target="_blank">PDF</a>' for u in arr if u.startswith("http"))
      except:
        return ""
    df["PDF"] = df["PDF"].apply(make_pdfs)

    # — wrap abstracts in <details>, hover shows full text
    def wrap_abstract(txt, n=100):
      if not isinstance(txt, str) or len(txt) <= n:
        return txt or ""
      summary = txt[:n].rsplit(" ",1)[0] + "…"
      esc = txt.replace('"', '&quot;')
      return (
        f'<details title="{esc}">'
        f'  <summary>{summary}</summary>'
        f'  <div style="padding:4px 0;">{txt}</div>'
        f'</details>'
      )
    df["Abstract"] = df["Abstract"].apply(wrap_abstract)

    st.write(f"🔍 Found {len(df)} of {len(df_all)} papers matching the filters.")
    csv = df_all.to_csv(index=False)
    st.download_button("Download current view as CSV", csv, "papers.csv")

    # ─── Convert to HTML & DataTables embed ──────────────────────
    html_table = df.to_html(
      index=False,
      escape=False,
      table_id="papers",
      classes="display stripe hover compact"
    )

    # — CSS to tame dark mode, fix sizes, show sort icons
    css = """
    <style>
    table.dataTable {
    color: #eee;
    background: #1c1c1e;
    border: 1px solid #444;
    table-layout: fixed;
    }
    table.dataTable th, table.dataTable td {
    padding: 8px 10px !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    }
    table.dataTable th {
    background: #2c2c2e !important;
    }
    table.dataTable th.sorting:after,
    table.dataTable th.sorting_asc:after,
    table.dataTable th.sorting_desc:after {
    color: #999;
    }
    table.dataTable tr {
    height: 48px;
    }
    table.compact th, table.compact td {
    padding: 4px 6px !important;
    }
    </style>
    """

    html_string = f"""
    <style>
    /* dark mode & fixed layout */
    table.dataTable {{
    color: #eee;
    background: #1c1c1e;
    border: 1px solid #444;
    table-layout: fixed;
    }}
    table.dataTable th, table.dataTable td {{
    padding: 8px 10px !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    }}
    table.dataTable th {{ background: #2c2c2e !important }}
    table.dataTable tr {{ height: 48px }}
    </style>

    <link 
    rel="stylesheet" 
    href="https://cdn.datatables.net/1.13.4/css/jquery.dataTables.min.css"/>
    <link 
    rel="stylesheet" 
    href="https://cdn.datatables.net/searchbuilder/1.4.2/css/searchBuilder.dataTables.min.css"/>

    <script src="https://code.jquery.com/jquery-3.5.1.js"></script>
    <script 
    src="https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js"></script>
    <script 
    src="https://cdn.datatables.net/searchbuilder/1.4.2/js/dataTables.searchBuilder.min.js"></script>

    {html_table}

    <script>
    $(document).ready(function() {{
    $('#papers').DataTable({{
        ordering: true,
        pageLength: 25,
        lengthMenu: [[25,50,100,-1],[25,50,100,'All']],
        autoWidth: false,
        scrollX: true,
        dom: 'QBfrtip',            // Q = SearchBuilder
        searchBuilder: {{
        columns: [1,2,4,5]       // apply filter UI to Genre, Authors, Publisher, Journal
        }}
    }});
    }});
    </script>
    """

    st.components.v1.html(html_string, height=700, scrolling=True)


