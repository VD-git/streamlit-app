import os
from mistralai import Mistral
import streamlit as st
import pymongo
import pandas as pd
from io import StringIO
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import AzureOpenAI
from tenacity import (retry, stop_after_attempt, wait_random_exponential)


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

class ChatEmbeddings:
    def __init__(self):
        self.client = chromadb.CloudClient(
            api_key=st.secrets["chromadb"]["api_key"],
            tenant=st.secrets["chromadb"]["tenant"],
            database=st.secrets["chromadb"]["database"]
        )

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

    def upload_data(self, summary_text, collection_name = "my_collection"):
        self.collection = self.client.get_or_create_collection(
                name = collection_name,
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

        self.collection.add(ids=ids, documents=contents, metadatas=metadatas)


class ChatbotOpenAI:
    def __init__(self, context):
        self.api_key = st.secrets["azure"]["api_key"]
        self.model = "gpt-4o-mini"
        self.client = AzureOpenAI(
            api_version=st.secrets["azure"]["api_version"],
            azure_endpoint=st.secrets["azure"]["azure_endpoint"],
            api_key=self.api_key,
        )
        self.system_message = [
            {
                "role": "system", 
                "content": f"""
                You are an AI Assistant to reply based o the information that it were provided between triple bracks, in case you werent able to find a good answer, just reply that 'Sorry, I could not find anything about'
                Context: ```{context}```
                """
            }
        ]

        self.history_messages = []

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def make_question(self, question:str):
        print(question)
        if len(self.history_messages) > 0:
            messages = self.system_message + self.history_messages + [{"role": "user", "content": question}]
            response = self.client.chat.completions.create(model = self.model, messages = messages)
            content = response.choices[0].message.content
            self.history_messages += [{"role": "user", "content": question}, {"role": "assistant", "content": content if content is not None else 'Vazio'}]
            self.response = response
            return content
        else:
            messages = self.system_message + [{"role": "user", "content": question}]
            response = self.client.chat.completions.create(model = self.model, messages = messages)
            content = response.choices[0].message.content
            self.history_messages += [{"role": "user", "content": question}, {"role": "assistant", "content": content if content is not None else 'Vazio'}]
            self.response = response
            return content
