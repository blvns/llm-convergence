import json
import os

original_path = "LLMs/corpora/dailydialog_corpus/dev/outputs/gemma-3-1b-it/evaluation/outputs.json"
converted_path = "stylometrics/data/movie_corpus/gemma-3-1b-it/converted_original.json"

# Check that file exists
print("Checking file existence...")
if not os.path.exists(original_path):
    print("❌ File not found:", original_path)
    exit()

# Load raw data
print("Loading raw JSON...")
with open(original_path, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print("Raw input conversations:", len(raw_data))

# Convert to list format
converted = []
for convo_id, utterances in raw_data.items():
    converted.append({
        "conversation_id": convo_id,
        "utterances": utterances
    })

# Save converted version
with open(converted_path, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=4, ensure_ascii=False)

# Load converted again and count
with open(converted_path, "r", encoding="utf-8") as f:
    converted_data = json.load(f)

print("Converted conversations:", len(converted_data))
print("✅ Script started.")

import json
import os

original_path = "LLMs/corpora/dailydialog_corpus/dev/outputs/gemma-3-1b-it/evaluation/outputs.json"
converted_path = "stylometrics/data/movie_corpus/gemma-3-1b-it/test_converted.json"

print("📄 Checking file existence...")

if not os.path.exists(original_path):
    print("❌ File not found:", original_path)
    exit()

print("📥 Loading JSON...")
with open(original_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("✅ Loaded. Conversation count:", len(data))
