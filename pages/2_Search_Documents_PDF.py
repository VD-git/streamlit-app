import streamlit as st
import logging
from logging import getLogger
import uuid
import time

from datetime import datetime, timedelta
from utils import Embeddings
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
    embeddings = Embeddings()

    client = chromadb.CloudClient(
        api_key=st.secrets["chromadb"]["api_key"],
        tenant=st.secrets["chromadb"]["tenant"],
        database=st.secrets["chromadb"]["database"]
    )
    
    if uploaded_file is not None:

        num_pages, summary_text = embeddings.read_pdf(uploaded_file)
        st.write(f"Number of pages: {num_pages}")
        # st.write(summary_text)

        collection = client.get_or_create_collection(
                name = "my_collection",
                embedding_function = OpenAIEmbeddingFunction(
                    model_name = "text-embedding-3-small",
                    api_key=st.secrets["azure"]["api_key"],
                    api_base =st.secrets["azure"]["azure_endpoint"],
                    api_type="azure",
                    api_version=st.secrets["azure"]["api_version"],
                    deployment_id = "text-embedding-3-small"
                )
        )

    
        ids = [i.get("id") for i in summary_text]
        contents = [i.get("content") for i in summary_text]
        metadatas = [i.get("metadata") for i in summary_text]

        collection.add(ids=ids, documents=contents, metadatas=metadatas)
        st.success("PDF uploaded and indexed. You can now ask questions.")
        input_disabled = False  # Enable input

    message = st.chat_input("Ask a question about the PDF", disabled=input_disabled)
    
    # Query after receiving user input
    if message and collection:
        result = collection.query(query_texts=[message], n_results=5)
        st.write("Top results:")
        for doc in result['documents'][0]:
            st.write(f"- {doc}")

        




