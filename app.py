import streamlit as st
import logging
from logging import getLogger
import pymongo

logger = getLogger()
if logger.handlers:  # logger is already setup, don't setup again
    pass
else:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    
def init_connection():
    client = pymongo.MongoClient(st.secrets["mongo"]["uri"])
    return client['Cluster0']

if __name__ == "__main__":
    dbname = init_connection()
    collection_name = dbname["user_1_items"]
    
    for it in collection_name.find():
        st.write(it)
    
    st.title('Stream Project')

    # st.write("Please log in to continue (username `test`, password `test`).")

    # username = st.text_input("Username")
    # password = st.text_input("Password", type="password")
    
    # if st.button("Log in", type="primary"):
    #     if username == "test" and password == "test":
    #         st.session_state.logged_in = True
    #         st.success("Logged in successfully!")
    #         st.write(st.session_state.logged_in)
    #     else:
    #         st.error("Incorrect username or password")

    st.write(f"Addr IP {st.context.headers.__dict__.get('_headers').get('X-Forwarded-For')}")

    logger.info(f"ip {st.context.headers.__dict__.get('_headers').get('X-Forwarded-For')}")




