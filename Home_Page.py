# Home.py
import streamlit as st

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


st.set_page_config(
    page_title="AI Projects Hub",
    page_icon="🤖",
)

st.title("Welcome to the AI Projects Hub")

st.markdown(
    """
    Explore a collection of AI-powered projects built using cutting-edge technologies.

    ## 🚀 Featured Projects

    **1. Chatbot With Mistral + MongoDB**  
    An interactive chatbot powered by the [Mistral API](https://mistral.ai/) (`mistral-small-latest`). This chatbot supports real-time conversations and uses [MongoDB](https://www.mongodb.com/) to store chat history securely.  
    > ⚠️ *Note: All messages exchanged with the chatbot are saved in the database.*
    --------------------------------------------------------------------------------------------------------------------------

    **2. Search Documents PDF**
    You can upload a PDF and, using the **RAG (Retrieval-Augmented Generation)** architecture, ask questions about its content.

    - **Similarity score**: Adjust how closely the answers match your query:
      - `0` → Only exact matches are considered. Some queries may return nothing.
      - `2` → Considers all possible matches.
      - **Default:** `1.5` → Balances accuracy and coverage.

    - The system uses **GPT-4o-mini** for chat, retrieving information from **ChromaDB**.
    > ⚠️ *Note: All uploaded PDFs are stored in the Chroma database. You can remove later your collection if needed.*
    --------------------------------------------------------------------------------------------------------------------------
    """
)

st.sidebar.success("👉 Select a project from the sidebar")
