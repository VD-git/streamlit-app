import streamlit as st
import logging
from logging import getLogger
import uuid
import tempfile
import time

from datetime import datetime, timedelta
from utils import RAGLangChain

logger = getLogger()
if logger.handlers:  # logger is already setup, don't setup again
    pass
else:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

if __name__ == "__main__":
    
    if "clear_chat" not in st.session_state:
        st.session_state.clear_chat = False

    input_disabled = True
    uploaded_file = st.file_uploader("Choose a file", type = ['pdf', 'csv', 'html'])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        rag_chain = RAGLangChain(file_name = uploaded_file.name, file_path = tmp_file_path)
        input_disabled = False
        st.session_state.clear_chat = True
        st.success(f"Document uploaded and indexed: `{uploaded_file.name}`")

    message = st.chat_input("Ask a question about the Document", disabled=input_disabled)
    
    if message:
        st.write(f"🙋 **You:** {message}")

        payload = rag_chain.invoke(message)
        response, faithfulness, context_precision, pages = payload.get("response"), payload.get("faithfulness"), payload.get("context_precision"), payload.get("pages")
        
        st.markdown(f"🤖 **Bot:** {response}")
        st.markdown(f"""<p style='font-size: 12px;'>
            Pages used: {", ".join([str(p) for p in list(set(pages))])}<br>
            Faithfulness Score: {round(faithfulness, 2)}<br>
            Context Precision Score: {round(context_precision, 2)}</p>""", unsafe_allow_html=True
        )

