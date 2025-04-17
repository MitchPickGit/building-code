import streamlit as st
import os
import io
import datetime
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# 🚪 Streamlit config
st.set_page_config(page_title="Building Regs Chatbot", page_icon="🏗️")
st.title("🏗️ Building Regulations Chatbot")

# 🔐 Prompt user for OpenAI API key
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key", type="password", help="You can get one at https://platform.openai.com/account/api-keys"
    )

if not st.session_state.openai_api_key:
    st.sidebar.warning("Please enter your OpenAI API key to continue.")
    st.stop()

# ✅ Setup LangChain + vectorstore
embedding = OpenAIEmbeddings(openai_api_key=st.session_state.openai_api_key)
vectorstore = FAISS.load_local("faiss_index", embedding)
retriever = vectorstore.as_retriever()

# ✅ Memory and chat chain
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=st.session_state.openai_api_key)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
)

# 📌 Init session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_sources" not in st.session_state:
    st.session_state.chat_sources = []

# 💬 User input
user_question = st.chat_input("Ask a question about the building regulations...")

if user_question:
    with st.spinner("Thinking..."):
        result = qa_chain({"question": user_question})
        answer = result["answer"]
        sources = result.get("source_documents", [])

        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Bot", answer))

        # 📄 Format citations
        citations = []
        for doc in sources:
            source = doc.metadata.get("source", "")
            clause = doc.metadata.get("clause", "")
            part = doc.metadata.get("part", "")
            division = doc.metadata.get("division", "")
            page = doc.metadata.get("page", "")
            url = doc.metadata.get("pdf_url", "")
            label = f"**{source}** – {clause} ({part}, {division}, Page {page}) [🔗 PDF]({url})"
            citations.append(label)

        formatted_refs = "\n\nSources:\n" + "\n".join(citations) if citations else ""
        st.session_state.chat_sources.append((formatted_refs, ""))

# 🧾 Chat display
for i, (sender, message) in enumerate(st.session_state.chat_history):
    if sender == "You":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
        refs, _ = st.session_state.chat_sources[i // 2] if i // 2 < len(st.session_state.chat_sources) else ("", "")
        if refs:
            with st.expander("🔎 Show Sources"):
                st.markdown(refs)

# 📁 Export button
def export_chat():
    buffer = io.StringIO()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    buffer.write(f"Building Regulations Chat Export ({now})\n\n")
    for i, (sender
