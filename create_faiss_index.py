import os
import json
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# ✅ Load .env file to get your OpenAI key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(f"🔑 DEBUG: API key = {api_key}")


if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Make sure .env is set properly.")

# ✅ Use correct local paths
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

# 🔄 Load both sources
reg_docs = load_documents(regulations_path)
act_docs = load_documents(act_path)
all_docs = reg_docs + act_docs

print(f"🔍 Loaded {len(all_docs)} total documents")

# 🧠 Generate FAISS index
embedding = OpenAIEmbeddings(openai_api_key=api_key)
db = FAISS.from_documents(all_docs, embedding)
db.save_local("faiss_index")

print("✅ FAISS index saved to faiss_index/")
