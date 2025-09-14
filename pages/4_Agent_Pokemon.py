import streamlit as st
import logging
from logging import getLogger
import uuid
import time

from datetime import datetime, timedelta
from utils import pokemon_images, init_connection, PokemonAgent

logger = getLogger()
if logger.handlers:  # logger is already setup, don't setup again
    pass
else:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

if __name__ == "__main__":
    dbname = init_connection()
    collection_name = dbname["pokemon"]
    
    st.title('Agent Pokemon ⚡🐭')
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.start_time = datetime.now() - timedelta(hours=0)
        
        st.session_state.payload = {
            "_id": st.session_state.session_id,
            "time": st.session_state.start_time,
            "messages": []
            }

        st.session_state.pokemon_call = PokemonAgent()
        logger.info(st.session_state.session_id)

    if st.session_state.pokemon_call.POKEMON is None:
        st.write("Describe your pokemon so we can find it :)")
    
    if st.session_state.pokemon_call.POKEMON is not None:
        imgs = pokemon_images(pokemon_name = st.session_state.pokemon_call.POKEMON, n = 9)
        cols = st.columns(3)
        for idx, img in enumerate(imgs):
            col = cols[idx % 3]
            with col:
                st.image(img, width = 100)

        st.write(f"We gotcha {st.session_state.pokemon_call.POKEMON}! 🎮")
        
    if "pending_message" not in st.session_state:
        st.session_state.pending_message = None
    
    # Load last messages replied
    if st.session_state.pokemon_call.POKEMON is None:
        for message in st.session_state.payload["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    if st.session_state.pokemon_call.POKEMON is None:
        input_disabled = st.session_state.pending_message is not None
    else:
        input_disabled = True
    message = st.chat_input("Enter message", disabled=input_disabled)
    
    if message and not input_disabled:
        st.session_state.pending_message = message
        st.rerun()
            
    if st.session_state.pending_message:
        # User
        st.chat_message("user").markdown(st.session_state.pending_message)
        st.session_state.payload["messages"].append({"role": "user", "content": st.session_state.pending_message})
        
        # Assistant
        time.sleep(1)
        response = st.session_state.pokemon_call.stream_memory_responses(st.session_state.pending_message)
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
        
        st.session_state.pending_message = None
        st.rerun()

        




