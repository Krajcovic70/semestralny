import requests
import json
import numpy as np
import pandas as pd
import time

openai_api_key = ""
if openai_api_key is None:
    raise ValueError("OpenAI API key is not set in environment variables.")


# Function to get embeddings for a given text
def get_embedding(text, retries=3, delay=5):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": "text-embedding-ada-002",
        "input": text
    }

    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                embedding = response.json()['data'][0]['embedding']
                return np.array(embedding)
            else:
                print(f"Error: {response.status_code}, {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}): {e}")
            time.sleep(delay)

    raise Exception(f"Failed to get embedding after {retries} attempts")


# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)
    return dot_product / (norm_embedding1 * norm_embedding2)


# Load dataset
df = pd.read_csv('mojtext.txt', sep='\t', header=None, names=['original', 'text1', 'text2'])

# Compute embeddings and similarities
embeddings_text1 = []
embeddings_text2 = []
similarities = []

print("Computing embeddings and similarities...")
for index, row in df.iterrows():
    print(f"Processing row {index + 1}/{len(df)}...")
    try:
        embedding1 = get_embedding(row['text1'])
        embedding2 = get_embedding(row['text2'])

        embeddings_text1.append(embedding1)
        embeddings_text2.append(embedding2)

        similarity = cosine_similarity(embedding1, embedding2)
        similarities.append(similarity)
    except Exception as e:
        print(f"Failed to process row {index + 1}: {e}")
        embeddings_text1.append(None)
        embeddings_text2.append(None)
        similarities.append(None)

print("Embeddings and similarities computed successfully!")

# Add embeddings and similarities to dataframe
df['embedding_text1'] = embeddings_text1
df['embedding_text2'] = embeddings_text2
df['similarity'] = similarities


# Save similarities to a separate text file
df[['similarity']].to_csv('similarities_mojtext.txt', sep='\t', index=False, header=False)

# Print summary output
print(
    f"Processed {len(df)} rows. Similarities saved to 'similarities_ai_sick_sk.txt' and original data with similarities saved to 'original_with_similarities.txt'.")
