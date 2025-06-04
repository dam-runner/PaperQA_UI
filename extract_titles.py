import json

def extract_titles(input_file='metadata_trimmed.json', output_file='title_summary.json'):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Collect pairs of actual_title and derived_title
    title_pairs = []
    for actual_title, metadata in data.items():
        derived_title = metadata.get("title", "")
        title_pairs.append({
            "actual_title": actual_title,
            "derived_title": derived_title
        })

    # Save to new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(title_pairs, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    extract_titles()
