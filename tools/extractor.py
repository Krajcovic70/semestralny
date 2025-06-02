import json

input_file = "../sts_benchmark_test.json"
output_file = "../statisticke testy/annotations_slovakbert.txt"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

annotations = data.get("similarity_scores") or data.get("annotations") or data.get("responses") or []

if annotations and isinstance(annotations[0], dict):
    values = [item.get("numerical_response") for item in annotations if "numerical_response" in item]
else:
    values = [float(x) for x in annotations if isinstance(x, (int, float))]

with open(output_file, "w", encoding="utf-8") as f:
    for val in values:
        f.write(f"{val:.3f}\n")

print(f"Uložených {len(values)} anotácií do: {output_file}")