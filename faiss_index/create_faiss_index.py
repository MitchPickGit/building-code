import os
import json
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# Load .env with OPENAI_API_KEY
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

regulations_path = "data/structured_building_regulations_2018.json"
act_path = "data/structured_building_act_1993.json"

def load_documents(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        content = item["text"]
        metadata = {
            "source": item.get("source", ""),
            "part": item.get("part", ""),
            "division": item.get("division", ""),
            "subdivision": item.get("subdivision", ""),
            "clause": item.get("clause", ""),
            "title": item.get("title", ""),
            "page": item.get("page", ""),
            "pdf_url": item.get("pdf_url", ""),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

# Load and combine
reg_docs = load_documents(regulations_path)
act_docs = load_documents(act_path)
all_docs = reg_docs + act_docs

print(f"🔍 Loaded {len(all_docs)} documents")

# Embed and save FAISS
embedding = OpenAIEmbeddings(openai_api_key=api_key)
db = FAISS.from_documents(all_docs, embedding)
db.save_local("faiss_index")

print("✅ FAISS index saved to faiss_index/")
