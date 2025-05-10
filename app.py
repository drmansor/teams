from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import os

# Initialize Flask app
app = Flask(__name__)

# Load and validate environment variables
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise EnvironmentError("Missing OPENAI_API_KEY environment variable")
openai.api_key = openai_key

# Load and prepare CSV data
# Ensure 'jordan_transactions.csv' is in the same directory as this script
df = pd.read_csv("jordan_transactions.csv")

# Convert each row to a natural language description
texts = []
for _, row in df.iterrows():
    date = row['transaction_date'].split()[0].replace('/', ' ')
    description = (
        f"On {date}, at {row['branch_name']} ({row['mall_name']}), "
        f"a {row['transaction_type']} transaction of {row['transaction_amount']:.3f} JOD "
        f"(including {row['tax_amount']:.3f} JOD tax) was recorded. "
        f"The status was {row['transaction_status']}."
    )
    texts.append(description)

# Embed the texts using a lightweight transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

# Build a FAISS index for efficient similarity search
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

@app.route("/ask", methods=["POST"])
def ask():
    """
    Expects JSON: {"question": "..."}
    Returns JSON: {"answer": "..."}
    """
    data = request.get_json(force=True)
    question = data.get("question")
    if not question:
        return jsonify({"error": "Missing question"}), 400

    # Embed the question and retrieve top 5 relevant records
    q_emb = model.encode([question])
    distances, indices = index.search(np.array(q_emb), k=5)
    context = "\n".join(texts[i] for i in indices[0])

    # Construct the prompt for the language model
    prompt = (
        "You are a helpful assistant answering questions about Jordanian transactions. "
        f"Use the following records as context:\n{context}\n\n"
        f"Answer the question: {question}"
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"answer": answer})

if __name__ == "__main__":
    # Render sets PORT automatically; default to 10000 if not provided
    port = int(os.getenv("PORT", 10000))
    app.run(host="0.0.0.0", port=port)