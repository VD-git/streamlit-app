import streamlit as st
import logging
from logging import getLogger
import uuid
import time

from datetime import datetime, timedelta
from utils import ChatbotOpenAI, ChatEmbeddings

logger = getLogger()
if logger.handlers:  # logger is already setup, don't setup again
    pass
else:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

if __name__ == "__main__":

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = ChatEmbeddings()

    if "selected_collection" not in st.session_state:
        st.session_state.selected_collection = None
    
    if "clear_chat" not in st.session_state:
        st.session_state.clear_chat = False

    embeddings = st.session_state.embeddings
    input_disabled = True
    uploaded_file = st.file_uploader("Choose a file")
    SIMILARITY_THRESHOLD = st.sidebar.slider(
        "🔍 Cosine Similarity Threshold (lower = more relevant)", 
        min_value=0.0, 
        max_value=2.0, 
        value=1.50, 
        step=0.01
    )

    if uploaded_file is not None:
        num_pages, summary_text = embeddings.read_pdf(uploaded_file)
        st.write(f"Number of pages: {num_pages}")
    
        # Use filename as collection name
        collection_name = uploaded_file.name
        embeddings.upload_data(summary_text)
    
        # Update session state
        st.session_state.selected_collection = collection_name
        st.session_state.clear_chat = True  # Clear messages after upload
    
        st.success(f"PDF uploaded and indexed into collection: `{collection_name}`")

    collection_list = embeddings.list_collections()
    collection_options = ["-- Select a collection --"] + collection_list
    default_index = collection_options.index(st.session_state.selected_collection) if st.session_state.selected_collection in collection_options else 0
    selected_option = st.selectbox("Select a collection", collection_options, index=default_index)

    if selected_option != "-- Select a collection --":
        embeddings.load_collection(selected_option)
        st.session_state.selected_collection = selected_option
        input_disabled = False

    if st.session_state.clear_chat:
        st.info("Chat cleared after PDF upload.")
        st.session_state.clear_chat = False

    with st.expander("⚠️ Delete a collection"):
        delete_collection = st.selectbox("Select collection to delete", collection_list, key="delete_selectbox")
        if st.button("Delete Selected Collection", type="primary"):
            if delete_collection:
                embeddings.delete_collection(delete_collection)
                st.success(f"Collection `{delete_collection}` deleted.")
                if st.session_state.selected_collection == delete_collection:
                    st.session_state.selected_collection = None
                    input_disabled = True
                st.rerun()

    message = st.chat_input("Ask a question about the PDF", disabled=input_disabled)
    
    # Query after receiving user input
    if message and embeddings.collection:
        st.write(f"🙋 **You:** {message}")
        query_result = embeddings.collection.query(query_texts=[message])
        documents = [doc for doc in query_result['documents'][0]]
        metadatas = [metadata['page'] for metadata in query_result['metadatas'][0]]
        distances = [distance for distance in query_result['distances'][0]]

        filtered_results = [
            (doc, meta, dist)
            for doc, meta, dist in zip(documents, metadatas, distances)
            if dist <= SIMILARITY_THRESHOLD
        ]
        
        context = ", ".join([document[0] for document in filtered_results])
        pages = ", ".join([str(pg+1) for pg in sorted([page[1] for page in filtered_results])])
        coi = ChatbotOpenAI(context=context)
        reply_back = coi.make_question(question=message)
        st.markdown(f"🤖 **Bot:**\n{reply_back}\n\n📄 **Pages used:** {pages}")
