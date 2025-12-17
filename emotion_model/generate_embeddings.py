import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load dataset
df = pd.read_csv("emotion_dataset.csv")

texts = df["text"].tolist()
labels = df["emotion"].tolist()

# Load sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = []

# Encode in batches to avoid memory issues
batch_size = 32

for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i+batch_size]
    batch_embeddings = model.encode(batch_texts, show_progress_bar=False)
    embeddings.append(batch_embeddings)

X = np.vstack(embeddings)
y = np.array(labels)

# Save to disk
np.save("X_embeddings.npy", X)
np.save("y_labels.npy", y)

print("Embeddings shape:", X.shape)
print("Labels shape:", y.shape)
print("Saved X_embeddings.npy and y_labels.npy")
