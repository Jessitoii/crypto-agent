import json 

with open("data/nexus_elite_dataset_v4.json", "r", encoding="utf-8") as f:
    data = json.load(f)

refined_data = []

for entry in data:
    entry["tp_pct"] = abs(entry["tp_pct"])
    refined_data.append(entry)

with open("data/nexus_elite_dataset_v5.json", "w", encoding="utf-8") as f:
    json.dump(refined_data, f, ensure_ascii=False, indent=2)