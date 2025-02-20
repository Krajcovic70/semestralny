import nlpcloud
import os
import json

# Setup API client
API_KEY = ''
client = nlpcloud.Client("paraphrase-multilingual-mpnet-base-v2", API_KEY, gpu=False)

# File paths
INPUT_FILE = 'stsbenchmark_sk.txt'  # Input file path
OUTPUT_FILE = f"{os.path.splitext(INPUT_FILE)[0]}_results_NLP.json"  # Output JSON file

# Function to load existing results from JSON
def load_existing_results(output_file):
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Main function
def main():
    # Load previously processed results
    results = load_existing_results(OUTPUT_FILE)
    processed_lines = len(results)

    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    batch = []  # To store a batch of results
    for idx, line in enumerate(lines[processed_lines:]):  # Start from unprocessed lines
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue  # Skip invalid lines

        # Sentences to compare
        sentence1 = parts[1]
        sentence2 = parts[2]

        try:
            # Call the semantic similarity method
            response = client.semantic_similarity([sentence1, sentence2])
            similarity_score = response.get("score")  # Extract the score

            # Append new result
            batch.append({
                "line_number": idx + processed_lines + 1,
                "sentence1": sentence1,
                "sentence2": sentence2,
                "similarity_score": similarity_score
            })

            # Write batch to JSON every 10 lines
            if len(batch) == 10:
                results.extend(batch)  # Add the batch to the results
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
                    json.dump(results, outfile, ensure_ascii=False, indent=4)
                batch = []  # Clear the batch
                print(f"Processed up to line {idx + processed_lines + 1}")
        except Exception as e:
            print(f"Exception on line {idx + processed_lines + 1}: {e}")
            break  # Stop the loop on error to allow restarting later

    # Write remaining lines in the batch
    if batch:
        results.extend(batch)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            json.dump(results, outfile, ensure_ascii=False, indent=4)
        print(f"Processed final batch up to line {processed_lines + len(batch)}")

if __name__ == '__main__':
    main()
