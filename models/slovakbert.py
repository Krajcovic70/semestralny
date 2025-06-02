import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity

def get_sentence_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
    sentence_embedding = torch.mean(embeddings, dim=1)
    return sentence_embedding.numpy()

#tokenizer = RobertaTokenizer.from_pretrained('gerulata/slovakbert')
#model = RobertaModel.from_pretrained('gerulata/slovakbert')


tokenizer = RobertaTokenizer.from_pretrained('../slovakbert-sts-model')
model = RobertaModel.from_pretrained('../slovakbert-sts-model')

dataset_path = '../stsbenchmark_sk.txt'
data = pd.read_csv(dataset_path, sep='\t', header=None, names=['score', 'sentence1', 'sentence2'])

results = []

for index, row in data.iterrows():
    embedding1 = get_sentence_embedding(row['sentence1'], tokenizer, model)
    embedding2 = get_sentence_embedding(row['sentence2'], tokenizer, model)
    similarity = cosine_similarity(embedding1, embedding2)[0][0] * 5
    results.append(f"{similarity}\n")
    print(f"Processed pair {index + 1}/{len(data)}")

with open('../slovakbert_stsbenchmark_trenovany.txt', 'w') as file:
    file.writelines(results)

print("Processing complete, results saved to trained_sts_benchmark.txt.")
