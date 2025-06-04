import json
import pandas as pd

def load_and_flatten(json_path):
    """
    1) Load JSON where top‐level is: { paper_title: { ... metadata ... } }.
    2) Turn it into a list of dicts; also keep the paper_title as its own field.
    3) Use pd.json_normalize to flatten nested 'other' and any other nested dicts.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for title, meta in data.items():
        # Insert the paper-title into the dict so we can track it if needed
        meta_copy = meta.copy()
        meta_copy['_paper_title'] = title
        records.append(meta_copy)

    # json_normalize will flatten nested dicts automatically,
    # using dot-notation for keys.  By default it stops at lists of scalars,
    # so things like "authors": ["A","B"] will remain a list.
    df = pd.json_normalize(records, sep='.')

    return df

def count_nonempty(series):
    """
    Return True for values we consider "has data".  Adjust these rules as desired.
    - None or NaN → False
    - Empty string → False
    - Empty list → False
    - Otherwise → True
    """
    # First, treat NaN/NA as missing
    s = series.dropna()

    def is_nonempty(x):
        if x is None:
            return False
        if isinstance(x, str) and x.strip() == "":
            return False
        if isinstance(x, (list, dict)) and len(x) == 0:
            return False
        return True

    # Count how many pass the “nonempty” test
    return s.apply(is_nonempty).sum()

def summarize_fields(df):
    """
    For each column in df, compute:
      total = len(df)
      n_filled = count_nonempty(df[col])
      n_empty = total - n_filled
      pct_filled = n_filled / total * 100
    Return a new DataFrame with columns: field, total, n_filled, n_empty, pct_filled
    """
    rows = []
    total = len(df)
    for col in df.columns:
        n_filled = count_nonempty(df[col])
        n_empty = total - n_filled
        pct = (n_filled / total * 100) if total > 0 else 0
        rows.append({
            'field': col,
            'total_entries': total,
            'n_filled': n_filled,
            'n_missing': n_empty,
            'pct_filled': pct
        })
    summary_df = pd.DataFrame(rows).sort_values('pct_filled', ascending=False)
    return summary_df

if __name__ == '__main__':
    # 1) Change this to whatever your file is called:
    JSON_FILE = 'metadata_trimmed.json'

    # 2) Load and flatten:
    df = load_and_flatten(JSON_FILE)

    # 3) Generate the per‐field summary:
    summary = summarize_fields(df)

    # 4) Show it (or write to CSV)
    pd.set_option('display.max_rows', None)  # so you can see every field if many
    print(summary.to_string(index=False))

    # (Optional) write to disk:
    summary.to_csv('field_completeness.csv', index=False)
