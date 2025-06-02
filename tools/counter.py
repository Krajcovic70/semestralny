from collections import Counter
import re
import json
import os


def count_first_column_occurrences(file_path):
    counts = Counter()

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t', 1)
            if len(parts) < 2:
                continue

            first_part = parts[0]

            if re.match(r'^\d+(\.\d+)?$', first_part):
                counts[first_part] += 1

    return counts


def main():
    file_path = "stsbenchmark_sk.txt"
    occurrences = count_first_column_occurrences(file_path)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = f"counted_occurrences_{base_name}.json"

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(dict(occurrences), json_file, indent=4, ensure_ascii=False)

    print(f"Výsledok bol uložený do {output_file}")


if __name__ == "__main__":
    main()