import json

with open('../train8.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

sentence1 = data["sentence1"]
sentence2 = data["sentence2"]
similarity_score = data["similarity_score"]

midpoint = len(sentence1) // 2

data_part1 = {
    "sentence1": sentence1[:midpoint],
    "sentence2": sentence2[:midpoint],
    "similarity_score": similarity_score[:midpoint]
}

data_part2 = {
    "sentence1": sentence1[midpoint:],
    "sentence2": sentence2[midpoint:],
    "similarity_score": similarity_score[midpoint:]
}

with open('../train8.json', 'w', encoding='utf-8') as file1:
    json.dump(data_part1, file1, ensure_ascii=False, indent=4)

with open('../train88.json', 'w', encoding='utf-8') as file2:
    json.dump(data_part2, file2, ensure_ascii=False, indent=4)

print("Rozdelenie dokončené. Súbory sú uložené ako train_part1.json a train_part5.json")