import json
import pandas as pd
import matplotlib.pyplot as plt

file_path = "../occurences/counted_occurrences_sick_sk.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

df = pd.DataFrame(list(data.items()), columns=["Value", "Count"])
df["Value"] = df["Value"].astype(float)


df_grouped = df.groupby("Value")["Count"].sum().reset_index()

plt.figure(figsize=(10, 5))
plt.bar(df_grouped["Value"], df_grouped["Count"], width=0.05)
plt.xlabel("Rozsah hodnôt podobnosti viet", fontsize=14)
plt.ylabel("Počet výskytov v danom datasete", fontsize=14)
plt.title("Frekvencie výskytu", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.savefig("graph_sick_final.png")
