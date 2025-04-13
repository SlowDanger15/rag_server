import faiss
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your documents (could be your resume, experience, etc.)
documents = [
    "Rohan is an AI/ML intern at SAP working on RAG agents.",
    "He has experience in LLM fine-tuning, vector DBs, and Langchain.",
    "Previously worked at Akamai, NeoDocto, and did academic research.",
    # Add more documents or load from files
]

# TF-IDF vectorizer as a simple embedder (replace with OpenAI if needed)
vectorizer = TfidfVectorizer()
embeddings = vectorizer.fit_transform(documents).toarray()

# Save the original docs
with open("model/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

# Build and save FAISS index
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings, dtype='float32'))
faiss.write_index(index, "model/index.faiss")

print("✅ FAISS index built and saved.")
