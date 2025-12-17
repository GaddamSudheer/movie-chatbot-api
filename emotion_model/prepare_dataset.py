from datasets import load_dataset
import pandas as pd
from label_mapping import GOEMOTION_MAP

# Load dataset
dataset = load_dataset("go_emotions", split="train")

label_names = dataset.features["labels"].feature.names

texts = []
final_labels = []

for row in dataset:
    text = row["text"]
    labels = row["labels"]

    # Keep only single-label samples
    if len(labels) != 1:
        continue

    original_label = label_names[labels[0]]

    # Map to final emotion
    if original_label not in GOEMOTION_MAP:
        continue

    mapped_label = GOEMOTION_MAP[original_label]

    texts.append(text)
    final_labels.append(mapped_label)

df = pd.DataFrame({
    "text": texts,
    "emotion": final_labels
})

print("Samples after cleaning:", len(df))
print(df["emotion"].value_counts())

# Save clean dataset
df.to_csv("emotion_dataset.csv", index=False)
print("Saved emotion_dataset.csv")
