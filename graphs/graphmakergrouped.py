import json
import pandas as pd
import matplotlib.pyplot as plt

# Load JSON data
file_path = "../occurences/counted_occurrences_stsbenchmark_sk.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# Convert to DataFrame
df = pd.DataFrame(list(data.items()), columns=["Value", "Count"])
df["Value"] = df["Value"].astype(float)

# Define fixed bins and labels
bins = [0, 1, 2, 3, 4, 5]
labels = ["<0,1>", "(1,2>", "(2,3>", "(3,4>", "(4,5>"]
df["Range"] = pd.cut(df["Value"], bins=bins, right=True, include_lowest=True, labels=labels)

# Group and sum counts by Range
df_grouped = df.groupby("Range")["Count"].sum().reset_index()

# Plotting
plt.figure(figsize=(10, 5))
plt.bar(df_grouped["Range"].astype(str), df_grouped["Count"])
plt.xlabel("Rozsah hodnôt podobnosti viet (intervaly)")
plt.ylabel("Počet výskytov v danom datasete")
plt.title("Frekvencie výskytu (zgrupené podľa rozsahov)")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save plot
plt.tight_layout()
plt.savefig("graph_sts_benchmark_grouped_fixed.png")
