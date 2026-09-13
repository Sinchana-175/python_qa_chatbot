import os
import streamlit as st
from dotenv import load_dotenv
from src.vector_db import QAEngine

# Load environment variables from .env file
load_dotenv()

st.set_page_config(page_title="Python RAG Chatbot", page_icon="🐍")

st.title("🐍 AI-Powered Python Assistant")
st.caption("A RAG-powered chatbot answering your Python questions from a curated knowledge base.")

@st.cache_resource
def load_engine():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in .env file. Please check your configuration.")
        st.stop()
    return QAEngine(api_key=api_key)

engine = load_engine()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_query := st.chat_input("Ask any Python question..."):
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("Thinking..."):
        response = engine.get_answer(user_query)

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
