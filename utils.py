import os
from mistralai import Mistral
import streamlit as st
import pymongo

def init_connection():
    """
    Function that makes the connection with Mongo, secrets are inserted directly into the streamlit interface
    """
    client = pymongo.MongoClient(st.secrets["mongo"]["uri"])
    return client['Cluster0']

class ChatbotMistral:
    def __init__(self):
        self.api_key = st.secrets["mistral"]["api_key"]
        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-small-latest"
        self.system_message = [{"role": "system", "content": "Você é um chatbot que responde o que perguntarem para você"}]
        self.history_messages = []

    def make_question(self, question:str):
        """
        Method to give a reply to the user
        Args:
            question (str): question made by the user
        Output:
            response (str): response by the chatbot
        """
        if len(self.history_messages) > 0:
            messages = self.system_message + self.history_messages + [{"role": "user", "content": question}]
            response = self.client.chat.complete(model = self.model, messages = messages).choices[0].message.content
            self.history_messages += [{"role": "user", "content": question}, {"role": "assistant", "content": response}]
            return response
        else:
            messages = self.system_message + [{"role": "user", "content": question}]
            response = self.client.chat.complete(model = self.model, messages = messages).choices[0].message.content
            self.history_messages = [{"role": "user", "content": question}, {"role": "assistant", "content": response}]
            return response