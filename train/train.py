import torch
import json
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

model_name = "gerulata/slovakbert"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression")  # 1 výstupná hodnota (skóre podobnosti)

train_file_path = "../sts_benchmark_train.json"
eval_file_path = "../sts_benchmark_train.json"
with open(train_file_path, "r", encoding="utf-8") as f:
    data_json = json.load(f)

data = pd.DataFrame({
    "sentence1": data_json["sentence1"],
    "sentence2": data_json["sentence2"],
    "score": data_json["similarity_score"]
})

eval_data_json = json.load(open(eval_file_path, "r", encoding="utf-8"))
eval_data = pd.DataFrame({
    "sentence1": eval_data_json["sentence1"],
    "sentence2": eval_data_json["sentence2"],
    "score": eval_data_json["similarity_score"]
})

# Funkcia na tokenizáciu textov
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = examples["score"]  # Pridanie skóre ako labels
    return tokenized_inputs
    tokenized_inputs = tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = examples["score"]  # Pridanie skóre ako labels
    return tokenized_inputs
    return tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=512)

data["score"] = data["score"].astype(float)
data["score"] = data["score"].astype(float)
dataset = Dataset.from_pandas(data)

eval_data["score"] = eval_data["score"].astype(float)
eval_data["score"] = eval_data["score"].astype(float)
eval_dataset = Dataset.from_pandas(eval_data).map(tokenize_function, batched=True)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

model.save_pretrained("fine_tuned_slovakbert")
tokenizer.save_pretrained("fine_tuned_slovakbert")

print("Tréning je dokončený.")
