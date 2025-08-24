# Home.py
import streamlit as st

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

    """
)

import sqlite3
st.write(sqlite3.sqlite_version)


st.sidebar.success("👉 Select a project from the sidebar")
