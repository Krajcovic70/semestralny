import json
import pandas as pd
import matplotlib.pyplot as plt

# Načítanie JSON súboru
file_path = "../occurences/counted_occurrences_stsbenchmark_sk.json"  # Uisti sa, že cesta k súboru je správna
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Konverzia do DataFrame
df = pd.DataFrame(list(data.items()), columns=["Value", "Count"])
df["Value"] = df["Value"].astype(float)


df_grouped = df.groupby("Value")["Count"].sum().reset_index()

# Vytvorenie grafu
plt.figure(figsize=(10, 5))
plt.bar(df_grouped["Value"], df_grouped["Count"], width=0.05)
plt.xlabel("Rozsah hodnôt podobnosti viet")
plt.ylabel("Počet výskytov v danom datasete")
plt.title("Frekvencie výskytu")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Uloženie grafu ako obrázok namiesto priameho zobrazovania
plt.savefig("graph_sts_benchmark.png")  # Uloží graf do súboru graph.png
