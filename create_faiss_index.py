# create_faiss_index.py

import json
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Load your OpenAI API key from environment
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load both structured JSON files
sources = [
    ("structured_building_regulations_2018.json", "Building Regulations 2018"),
    ("structured_building_act_1993.json", "Building Act 1993"),
]

all_docs = []

for file, source_name in sources:
    with open(file, "r", encoding="utf-8") as f:
        pages = json.load(f)
        for entry in pages:
            metadata = {
                "page": entry.get("page"),
                "part": entry.get("part", ""),
                "division": entry.get("division", ""),
                "citation": entry.get("citation", ""),
                "source": source_name
            }
            doc = Document(page_content=entry["text"], metadata=metadata)
            all_docs.append(doc)

# Create embeddings and FAISS index
print("🔍 Creating embeddings and saving FAISS index...")
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(all_docs, embedding)
db.save_local("faiss_index")
print("✅ FAISS index saved to: faiss_index/")
