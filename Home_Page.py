# Home.py
# removing error from asyncio plus ragas
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import streamlit as st
from utils import PokemonAgent

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

    You can upload a PDF and, using the **RAG (Retrieval Augmented Generation)** architecture, ask questions about its content.

    - **Similarity score**: Adjust how closely the answers match your query:
      - `0` → Only exact matches are considered. Some queries may return nothing.
      - `2` → Considers all possible matches.
      - **Default:** `1.5` → Balances accuracy and coverage.

    - The system uses **GPT-4o-mini** for chat, retrieving information from **ChromaDB**.
    > ⚠️ *Note: All uploaded PDFs are stored in the Chroma database. You can remove later your collection if needed.*
    --------------------------------------------------------------------------------------------------------------------------

    **3. RAG Langchain**

    This section you are able to upload a PDF, CSV or HTML, using **RAG (Retrieval Augmented Generation)**, ask question about its content.

    - Response will be query up to 3 chunks;
        - **Pages/Rows**:
            - For PDFs, the specific pages that were searched;
            - For CSVs, the rows that were considered;
            - For HTML, no page or row details are shown.
        - **Faithfulness Score**: Indicates how closely the generated response matches original content (0-1);
        - **Context Precision Score**: Shows how well the response aligns with retrieved context (0-1).
    - The system uses **GPT-4o-mini** for chat, retrieving information from **ChromaDB**.
    --------------------------------------------------------------------------------------------------------------------------

    **4. Agent Pokemon**

    This system uses a Langchain architecture with multi-agents, powered by the Azure API `gpt-4o-mini`. helps you in the search for your Pokémon, and once it's found, confirm with "yes" to receive its images. This chatbot supports real-time conversations and uses [MongoDB](https://www.mongodb.com/) to store chat history securely.
    The architecture consists of two primary nodes and one auxiliary node:

    - **chatbot:** Responsible for guessing the Pokémon based on user input;
    - **evaluator:** Validates whether the guessed Pokémon exists in the database;
    - **guide_node:** Provides better guidance and messaging to the user throughout the interaction.
    """
)

with st.expander("🔍 View Architecture Diagram of LangGraph"):
    st.image(PokemonAgent().graph_architecture(), caption="Architecture LangGraph", width=670)

st.markdown(
    """
    > ⚠️ *Note: All messages exchanged with the chatbot are saved in the database.*
    --------------------------------------------------------------------------------------------------------------------------
    """
)

st.sidebar.success("👉 Select a project from the sidebar")
