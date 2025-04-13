import os
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# ✅ Your OpenAI Key
os.environ["OPENAI_API_KEY"] = "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"

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
embedding = OpenAIEmbeddings()
db = FAISS.from_documents(all_docs, embedding)
db.save_local("faiss_index")

print("✅ FAISS index saved to faiss_index/")
