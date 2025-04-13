from sentence_transformers import SentenceTransformer
import faiss
import json
from flask import Flask, request

app = Flask(__name__)

# Load documents
docs = []
with open("documents/resume.txt", "r") as f:
    docs.append(f.read())
with open("documents/projects.txt", "r") as f:
    docs.append(f.read())

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(docs)
index = faiss.IndexFlatL2(len(doc_embeddings[0]))
index.add(doc_embeddings)

@app.route("/api/rag", methods=["POST"])
def rag_search():
    query = request.json["query"]
    query_vec = model.encode([query])
    _, I = index.search(query_vec, k=1)
    result = docs[I[0][0]]
    return json.dumps({"answer": result})
