from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import pandas as pd
import json

# 1. Load your data
with open("sts_benchmark_train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Create training examples
train_examples = []
for s1, s2, score in zip(data["sentence1"], data["sentence2"], data["similarity_score"]):
    train_examples.append(InputExample(texts=[s1, s2], label=float(score) / 5.0))  # normalize to 0-1

# 3. Define model (Gerulata SlovakBERT with mean pooling)
word_embedding_model = models.Transformer("gerulata/slovakbert")
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# 4. Prepare DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model=model)

# 5. Fine-tune the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path="./slovakbert-sts-model"
)
