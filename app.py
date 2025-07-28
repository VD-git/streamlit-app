import streamlit as st
import logging
from logging import getLogger
import uuid

from datetime import datetime, timedelta
from utils import ChatbotMistral, init_connection

logger = getLogger()
if logger.handlers:  # logger is already setup, don't setup again
    pass
else:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

if __name__ == "__main__":
    dbname = init_connection()
    collection_name = dbname["streamlit"]
    cm = ChatbotMistral()
    
    st.title('Chatbot Mistral + Mongo Project')
    
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.start_time = datetime.now() - timedelta(hours=3)
        
        st.session_state.payload = {
            "_id": st.session_state.session_id,
            "time": st.session_state.start_time,
            "messages": []
            }
        
    logger.info(st.session_state.session_id)
    
    # Load last messages replied
    for message in st.session_state.payload["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    message = st.chat_input("Enter message")

    if message:
        # User
        st.chat_message("user").markdown(message)
        st.session_state.payload["messages"].append({"role": "user", "content": message})
        
        # Assistant
        response = cm.make_question(message)
        st.session_state.payload["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Mongo
        logger.info(st.session_state.payload)
        result = collection_name.replace_one(
            {"_id": st.session_state.session_id},
            st.session_state.payload,
            upsert = True
            )
        logger.info(result)
        




