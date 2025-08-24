import os
from mistralai import Mistral
import streamlit as st
import pymongo
import pandas as pd
from io import StringIO
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

class Embeddings:
    def __init__(self):
        pass

    def read_pdf(self, uploaded_file):
        summary_text = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)

        with uploaded_file as file:
            reader = pypdf.PdfReader(file)

            num_pages = len(reader.pages)
        
            for page in range(num_pages):
                page_read = reader.pages[page]
                chuncks = text_splitter.split_text(page_read.extract_text())
                for chunck in range(len(chuncks)):
                    summary_text.append({"id": "_".join([str(page), str(chunck)]), "content": chuncks[chunck], "metadata": {"page": page, "chunck": chunck}})
        return num_pages, summary_text


    # uploaded_file = st.file_uploader("Choose a file")
    # if uploaded_file is not None:
    #     # To read file as bytes:
    #     bytes_data = uploaded_file.getvalue()
    #     st.write(bytes_data)
    
    #     # To convert to a string based IO:
    #     stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #     st.write(stringio)
    
    #     # To read file as string:
    #     string_data = stringio.read()
    #     st.write(string_data)
    
    #     # Open the PDF file in binary read mode
    #     with open(string_data, 'rb') as file:
    #         # Create a PdfReader object
    #         reader = pypdf.PdfReader(file)
        
    #         # Get the number of pages
    #         num_pages = len(reader.pages)
    #         print(f"Number of pages: {num_pages}")
        
    #         # Extract text from a specific page (e.g., the first page)
    #         text = []
    #         for n in range(num_pages):
    #             first_page = reader.pages[n]
    #             text.append(first_page.extract_text())
    #     st.write(text)