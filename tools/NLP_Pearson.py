import json
import numpy as np
from scipy.stats import pearsonr

# File paths
JSON_FILE = "stsbenchmark_sk_results_NLP.json"  # JSON file with similarity scores
TEXT_FILE = "stsbenchmark_sk.txt"  # Text file with initial numbers
OUTPUT_FILE = "stsbenchmark_sk_pearson_results_NLP.txt"  # File to save Pearson coefficient


def load_json(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []


def load_text_numbers(file_path):
    """Loads numbers from the beginning of lines in a text file."""
    numbers = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts and parts[0].replace('.', '', 1).isdigit():
                    numbers.append(float(parts[0]))
        return numbers
    except Exception as e:
        print(f"Error loading text file: {e}")
        return []


def calculate_pearson(json_values, text_values):
    """Calculates Pearson correlation coefficient."""
    if len(json_values) != len(text_values):
        print("Warning: JSON and text values lengths do not match!")
        min_len = min(len(json_values), len(text_values))
        json_values = json_values[:min_len]
        text_values = text_values[:min_len]

    # Calculate Pearson coefficient
    coefficient, _ = pearsonr(json_values, text_values)
    return coefficient


def main():
    # Load data
    json_data = load_json(JSON_FILE)
    text_numbers = load_text_numbers(TEXT_FILE)

    # Process JSON scores
    json_values = [entry["similarity_score"] * 5 for entry in json_data]

    # Calculate Pearson coefficient
    pearson_coefficient = calculate_pearson(json_values, text_numbers)

    # Save the result
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"Pearson correlation coefficient: {pearson_coefficient:.5f}\n")
        print(f"Pearson coefficient saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error saving Pearson coefficient: {e}")


if __name__ == "__main__":
    main()
