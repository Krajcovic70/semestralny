import nlpcloud
import csv

client = nlpcloud.Client("paraphrase-multilingual-mpnet-base-v2", "", gpu=False)

def process_dataset(input_file, output_file):
    results = []

    with open(input_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) >= 3:
                text1 = row[1]
                text2 = row[2]
                response = client.semantic_similarity([text1, text2])
                similarity_score = response['similarities'][0][1]
                results.append(f"Similarity between '{text1}' and '{text2}': {similarity_score}")

    with open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(result + "\n")

input_file = 'test.txt'
output_file = 'testout.txt'
process_dataset(input_file, output_file)
