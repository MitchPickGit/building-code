import os
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# === CONFIG ===
ACT_PATH = "data/structured_building_act_1993.json"
REGS_PATH = "data/structured_building_regulations_2018.json"
INDEX_DIR = "faiss_index"

# === Load JSON Files ===
with open(ACT_PATH, "r", encoding="utf-8") as f:
    act_data = json.load(f)

with open(REGS_PATH, "r", encoding="utf-8") as f:
    regs_data = json.load(f)

combined_data = act_data + regs_data

# === Convert to LangChain Documents ===
documents = [
    Document(
        page_content=item["text"],
        metadata={
            "source": item.get("source", ""),
            "part": item.get("part", ""),
            "division": item.get("division", ""),
            "subdivision": item.get("subdivision", ""),
            "section": item.get("section", ""),
            "clause": item.get("clause", ""),
            "page": item.get("page", "")
        }
    )
    for item in combined_data
]

# === Embed and Save Vector Index ===
print("\n🔍 Creating embeddings and saving FAISS index...")
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local(INDEX_DIR)
print(f"✅ FAISS index saved to: {INDEX_DIR}/")

