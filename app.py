import streamlit as st
import logging
from logging import getLogger

logger = getLogger()
if logger.handlers:  # logger is already setup, don't setup again
    pass
else:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    



if __name__ == "__main__":
    
    st.title('Stream Project')

    st.write(f"Addr IP {st.context.headers.__dict__.get('_headers').get('X-Forwarded-For')}")

    logger.info(f"ip {st.context.headers.__dict__.get('_headers').get('X-Forwarded-For')}")




