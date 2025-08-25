import streamlit as st
import logging
from logging import getLogger
import uuid
import time

from datetime import datetime, timedelta
from utils import ChatEmbeddings
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

logger = getLogger()
if logger.handlers:  # logger is already setup, don't setup again
    pass
else:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

if __name__ == "__main__":
    uploaded_file = st.file_uploader("Choose a file")
    input_disabled = True
    embeddings = ChatEmbeddings()
    
    if uploaded_file is not None:

        num_pages, summary_text = embeddings.read_pdf(uploaded_file)
        st.write(f"Number of pages: {num_pages}")

        embeddings.upload_data(summary_text)

        st.success("PDF uploaded and indexed. You can now ask questions.")
        input_disabled = False  # Enable input

    message = st.chat_input("Ask a question about the PDF", disabled=input_disabled)
    
    # Query after receiving user input
    if message:
        result = embeddings.collection.query(query_texts=[message], n_results=5)
        st.write("Top results:")
        for doc in result['documents'][0]:
            st.write(f"- {doc}")

        




