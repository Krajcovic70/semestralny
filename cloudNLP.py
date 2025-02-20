import nlpcloud
import csv

# Initialize the NLP Cloud client
client = nlpcloud.Client("paraphrase-multilingual-mpnet-base-v2", "", gpu=False)

# Function to process the dataset and save the outputs
def process_dataset(input_file, output_file):
    results = []

    # Read the input file
    with open(input_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) >= 3:
                text1 = row[1]
                text2 = row[2]
                # Compare the second and third columns using NLP Cloud
                response = client.semantic_similarity([text1, text2])
                similarity_score = response['similarities'][0][1]  # Assuming the response structure
                results.append(f"Similarity between '{text1}' and '{text2}': {similarity_score}")

    # Write the results to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(result + "\n")

# Example usage
input_file = 'test.txt'
output_file = 'testout.txt'
process_dataset(input_file, output_file)
