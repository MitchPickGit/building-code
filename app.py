import streamlit as st
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import openai
import io
import datetime

st.set_page_config(page_title="Building Regs Chatbot", page_icon="🏗️")
st.title("🏗️ Building Regulations Chatbot")

# Try to load API key from secrets, fallback to user input
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

if api_key:
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key

    # Load FAISS index from disk
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local("faiss_index", embeddings)
    retriever = vectorstore.as_retriever()

    # Memory and model
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

    # Session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_sources" not in st.session_state:
        st.session_state.chat_sources = []

    # Input
    user_question = st.chat_input("Ask a question about the building regulations...")

    if user_question:
        with st.spinner("Thinking..."):
            result = qa_chain({"question": user_question})
            answer = result["answer"]
            sources = result.get("source_documents", [])

            # Append user + bot message to history
            st.session_state.chat_history.append(("You", user_question))
            st.session_state.chat_history.append(("Bot", answer))

            # Append sources
            if sources:
                refs = set(doc.metadata.get("citation") for doc in sources)
                clause_texts = [f"\n\n**{doc.metadata['citation']}**\n{doc.metadata['text']}" for doc in sources]
                full_refs = "\n\nSources: " + ", ".join(sorted(refs))
                full_text = "\n".join(clause_texts)
                st.session_state.chat_sources.append((full_refs, full_text))
            else:
                st.session_state.chat_sources.append(("", ""))

    # Display chat history
    for i, (sender, message) in enumerate(st.session_state.chat_history):
        if sender == "You":
            st.chat_message("user").write(message)
        else:
            st.chat_message("assistant").write(message)
            # Show reference toggle
            refs, full_clause = st.session_state.chat_sources[i // 2] if i // 2 < len(st.session_state.chat_sources) else ("", "")
            if refs:
                with st.expander("🔎 Show Sources and Clause Text"):
                    st.markdown(refs)
                    st.markdown(full_clause)

    # Export chat log
    def export_chat():
        buffer = io.StringIO()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        buffer.write(f"Building Regulations Chat Export ({now})\n\n")
        for i, (sender, message) in enumerate(st.session_state.chat_history):
            buffer.write(f"{sender}: {message}\n")
            if sender == "Bot" and i // 2 < len(st.session_state.chat_sources):
                refs, full_clause = st.session_state.chat_sources[i // 2]
                buffer.write(f"{refs}\n\n")
        buffer.seek(0)
        return buffer

    st.sidebar.markdown("---")
    if st.sidebar.button("📁 Export Chat Log"):
        st.sidebar.download_button(
            label="Download Chat with References",
            data=export_chat(),
            file_name="building_regs_chat_log.txt",
            mime="text/plain"
        )

    st.sidebar.markdown("Built for contextual legal compliance and clear clause traceability.")

else:
    st.warning("Please enter your OpenAI API key to begin.")
