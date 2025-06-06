# === In papers_dashboard/app.py ===

import streamlit as st
import pandas as pd
import sqlite3
import json
import matplotlib.pyplot as plt

# ——————————————————————————————————————————
# 1) Database connection & global variables
# ——————————————————————————————————————————
DB_PATH = "asr_papers.db"

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
    df_cite = pd.read_sql("SELECT citation_count FROM papers;", conn)
    missing_cite = df_cite["citation_count"].isna().sum()
    valid_cite = df_cite["citation_count"].dropna().astype(int)

    st.subheader("Citation Count Distribution")
    cite_counts = valid_cite.value_counts().sort_index()
    fig_cite, ax_cite = plt.subplots()
    ax_cite.bar(cite_counts.index.astype(str), cite_counts.values)
    ax_cite.set_xlabel("Citation Count")
    ax_cite.set_ylabel("# of Papers")
    ax_cite.set_xticks(range(len(cite_counts)))
    ax_cite.set_xticklabels(cite_counts.index.astype(str), rotation=45, ha="right")
    st.pyplot(fig_cite)
    st.caption(f"⚠️ {missing_cite} of {total_papers} papers have unknown citation counts.")

    # 4.3) Papers per Author (Top 10)
    df_auth = pd.read_sql("SELECT authors FROM papers;", conn)
    all_authors = []
    for a in df_auth["authors"]:
        if a and a.strip():
            for name in a.split(";"):
                name_clean = name.strip()
                if name_clean:
                    all_authors.append(name_clean)
        else:
            all_authors.append("Unknown")
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

    # 4.4) Papers by Genre
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

    # 4.5) Papers by Journal (Top 10)
    df_j_raw = pd.read_sql("SELECT journal FROM papers;", conn)
    normalized_j = df_j_raw["journal"].apply(lambda j: normalize_journal(j))
    journal_counts = normalized_j.value_counts().head(10)
    missing_journal = (df_j_raw["journal"].isna() | (df_j_raw["journal"].str.strip() == "")).sum()

    st.subheader("Papers by Journal (Top 10)")
    fig_j, ax_j = plt.subplots()
    ax_j.pie(
        journal_counts.values,
        labels=journal_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax_j.axis("equal")
    st.pyplot(fig_j)
    st.caption(f"⚠️ {missing_journal} of {total_papers} papers have unknown or rare journals.")



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
    df_cite = pd.read_sql("SELECT citation_count FROM papers;", conn)
    missing_cite = df_cite["citation_count"].isna().sum()
    valid_cite = df_cite["citation_count"].dropna().astype(int)
    counts_cite = valid_cite.value_counts().sort_index()
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
    all_authors = []
    for a in df_auth["authors"]:
        if a and a.strip():
            for name in a.split(";"):
                name_clean = name.strip()
                if name_clean:
                    all_authors.append(name_clean)
        else:
            all_authors.append("Unknown")
    author_counts = pd.Series(all_authors).value_counts().head(10)
    fig_auth2, ax_auth2 = plt.subplots()
    ax_auth2.barh(author_counts.index, author_counts.values)
    ax_auth2.invert_yaxis()
    ax_auth2.set_xlabel("Number of Papers")
    ax_auth2.set_ylabel("Author")
    st.pyplot(fig_auth2)
    st.caption("Top 10 authors by paper count.")

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

    # — Sidebar Filters —
    st.sidebar.header("Filter Papers")

    # (1) Year filter — slider
    year_min = int(pd.read_sql("SELECT MIN(year) AS min_y FROM papers;", conn)["min_y"].iloc[0] or 2015)
    year_max = int(pd.read_sql("SELECT MAX(year) AS max_y FROM papers;", conn)["max_y"].iloc[0] or 2025)
    selected_years = st.sidebar.slider(
        "Publication Year Range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max)
    )

    # (2) Genre filter — multi-select
    raw_genres = pd.read_sql("SELECT DISTINCT genre FROM papers;", conn)["genre"].fillna("").tolist()
    norm_genres = sorted({normalize_genre(g) for g in raw_genres})
    selected_genres = st.sidebar.multiselect(
        "Genre",
        options=norm_genres,
        default=norm_genres
    )

    # (3) Journal filter — multi-select
    raw_journals = pd.read_sql("SELECT DISTINCT journal FROM papers;", conn)["journal"].fillna("").tolist()
    norm_journals = sorted({normalize_journal(j) for j in raw_journals})
    selected_journals = st.sidebar.multiselect(
        "Journal (Top 20 by frequency)",
        options=norm_journals,
        default=norm_journals
    )

    # (4) Keyword search in abstract
    search_kw = st.sidebar.text_input("Search abstracts for keyword")

    # — Build SQL WHERE clause for years only —
    query = "SELECT * FROM papers WHERE year BETWEEN ? AND ?"
    params = [selected_years[0], selected_years[1]]
    df_all = pd.read_sql(query, conn, params=params)

    # — Apply normalization & Python-level filters —
    df_all["genre_norm"] = df_all["genre"].apply(lambda g: normalize_genre(g))
    df_all["journal_norm"] = df_all["journal"].apply(lambda j: normalize_journal(j))

    # Filter by genre
    if selected_genres and set(selected_genres) != set(norm_genres):
        df_all = df_all[df_all["genre_norm"].isin(selected_genres)]

    # Filter by journal
    if selected_journals and set(selected_journals) != set(norm_journals):
        df_all = df_all[df_all["journal_norm"].isin(selected_journals)]

    # Filter by abstract keyword
    if search_kw and search_kw.strip():
        kw = search_kw.strip().lower()
        df_all = df_all[df_all["abstract"].fillna("").str.lower().str.contains(kw)]

    # — Prepare the DataFrame for display —
    def format_attachments(la):
        if not la or la.strip() == "" or la.strip() == "[]":
            return "Unknown"
        try:
            arr = json.loads(la)
        except:
            return "Unknown"
        links = []
        for u in arr:
            if u and u.startswith("http"):
                links.append(f"[PDF]({u})")
        return ", ".join(links) if links else "Unknown"

    def make_clickable(url):
        if url and url.strip().startswith("http"):
            return f"[Link]({url})"
        return "Unknown"

    # Choose the columns to display, in the order you want
    display_cols = [
        "title", "genre_norm", "authors", "year", "url",
        "publisher", "journal_norm", "doi", "abstract",
        "doi_url", "formatted_citation", "citation_count",
        "link_attachments"
    ]
    df_display = df_all[display_cols].copy()
    df_display.rename(columns={
        "title": "Title",
        "genre_norm": "Genre",
        "authors": "Authors",
        "year": "Year",
        "url": "URL",
        "publisher": "Publisher",
        "journal_norm": "Journal",
        "doi": "DOI",
        "abstract": "Abstract",
        "doi_url": "DOI_URL",
        "formatted_citation": "Citation",
        "citation_count": "Citation_Count",
        "link_attachments": "Link_Attachments"
    }, inplace=True)

    df_display = df_display.fillna("Unknown")
    df_display["URL"] = df_display["URL"].apply(make_clickable)
    df_display["Link_Attachments"] = df_display["Link_Attachments"].apply(format_attachments)

    st.write(f"🔍 Found {len(df_display)} papers matching the selected filters:")
    st.write("**Click “Link” or “PDF” to open the paper if available.**")
    st.dataframe(df_display, use_container_width=True)

    matched = len(df_display)
    caption_msg = f"{matched} of {total_papers} papers match your criteria."
    if search_kw:
        excluded = (df_all.shape[0] - matched)
        caption_msg += f" ({excluded} excluded because they had no match in the abstract.)"
    st.caption(caption_msg)


