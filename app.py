
import streamlit as st
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
import os

st.set_page_config(page_title="Building Regs Chatbot", page_icon="🏗️")
st.title("🏗️ Building Regulations Chatbot")

# --- API Key input ---
openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Load CSV
    df = pd.read_csv("structured_regulations.csv")

    # Preprocess data into documents
    texts = []
    metadatas = []
    for _, row in df.iterrows():
        content = row["Text"]
        if pd.notna(content) and content.strip():
            ref = row["Full Reference"]
            text = f"{ref}\n{content}"
            texts.append(text)
            metadatas.append({"reference": ref, "page": row["Start Page"]})

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents(texts, metadatas=metadatas)

    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Input box
    user_question = st.text_input("Ask a question about the building regulations:")

    if user_question:
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_question)
        st.markdown("### 💬 Answer:")
        st.write(response)
else:
    st.info("Please enter your OpenAI API key to start.")
