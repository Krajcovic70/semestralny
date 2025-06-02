import json
import pandas as pd
import matplotlib.pyplot as plt

file_path = "../occurences/counted_occurrences_sick_sk.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

df = pd.DataFrame(list(data.items()), columns=["Value", "Count"])
df["Value"] = df["Value"].astype(float)

bins = [0, 1, 2, 3, 4, 5]
labels = ["<0,1>", "(1,2>", "(2,3>", "(3,4>", "(4,5>"]
df["Range"] = pd.cut(df["Value"], bins=bins, right=True, include_lowest=True, labels=labels)

df_grouped = df.groupby("Range")["Count"].sum().reset_index()

plt.figure(figsize=(10, 5))
plt.bar(df_grouped["Range"].astype(str), df_grouped["Count"])
plt.xlabel("Rozsah hodnôt podobnosti viet (intervaly)", fontsize=14)
plt.ylabel("Počet výskytov v danom datasete", fontsize=14)
plt.title("Frekvencie výskytu (zgrupené podľa rozsahov)", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("graph_sick_grouped_final.png")
plt.show()
