import os
from mistralai import Mistral
import streamlit as st
import pymongo
import pandas as pd
from io import StringIO
import pypdf
import tempfile
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import AzureOpenAI
from tenacity import (retry, stop_after_attempt, wait_random_exponential)
import re
import unicodedata

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, CharacterTextSplitter
)
from langchain_chroma import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import (AzureOpenAIEmbeddings, AzureChatOpenAI)
from langchain_community.document_loaders import (PyPDFLoader, UnstructuredHTMLLoader)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langchain.schema import (
    HumanMessage, SystemMessage, AIMessage
)

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from ragas.integrations.langchain import EvaluatorChain
from ragas.metrics import context_precision, faithfulness

def init_connection():
    """
    Function that makes the connection with Mongo.
    Secrets are inserted directly into the streamlit interface
    """
    client = pymongo.MongoClient(st.secrets["mongo"]["uri"])
    return client['Cluster0']


class ChatbotMistral:
    def __init__(self):
        self.api_key = st.secrets["mistral"]["api_key"]
        self.client = Mistral(api_key=self.api_key)
        self.model = "mistral-small-latest"
        self.system_message = [
            {
                "role": "system",
                "content": (
                    "Você é um chatbot "
                    "que responde o que perguntarem para você."
                )
            }
        ]
        self.history_messages = []

    def make_question(self, question: str):
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
        self.name = None
        self.collection_list = [col.name for col in self.client.list_collections()]

    def list_collections(self):
        self.collection_list = [col.name for col in self.client.list_collections()]
        return self.collection_list

    def read_pdf(self, uploaded_file):
        summary_text = []
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=30)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
        )

        with uploaded_file as file:
            reader = pypdf.PdfReader(file)

            name = uploaded_file.name
            name = name.replace(' ', '_')
            name = unicodedata.normalize('NFKD', name)
            
            self.name = re.sub(r'[^\w._]', '', name)

            num_pages = len(reader.pages)
        
            for page in range(num_pages):
                page_read = reader.pages[page]
                chuncks = text_splitter.split_text(page_read.extract_text())
                for chunck in range(len(chuncks)):
                    summary_text.append({"id": "_".join([str(page), str(chunck)]), "content": chuncks[chunck], "metadata": {"page": page, "chunck": chunck}})
        return num_pages, summary_text

    def upload_data(self, summary_text):
        # self.client.delete_collection(name="my_collection")
        self.collection = self.client.get_or_create_collection(
                name = self.name,
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

    def load_collection(self, collection_name):
        self.collection = self.client.get_collection(
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

    def delete_collection(self, collection_name: str):
        try:
            self.client.delete_collection(name=collection_name)
            if self.collection_name == collection_name:
                self.collection = None
                self.collection_name = None
        except Exception as e:
            print(f"Error deleting collection '{collection_name}': {e}")

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

class RAGLangChain:
    def __init__(self, file_name, file_path, chunk_size = 100, chunk_overlap = 20):
        self.separators = ["\n\n", "\n", " ", ""]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_extension = self._define_file_extension(file_name)
        self.file_name = file_name
        self.file_path = file_path
        self._load_file()
        self.split_file(split_type = 'semantic')
        self._load_embedding_function()
        self._load_vector_store()
        self._load_language_model()
        self._load_template()
        self._load_chain()
        self._load_evaluators()
    def _define_file_extension(self, file_name: str):
        extension = file_name.split('.')[-1]
        assert extension in ['pdf', 'csv', 'html'], 'Only csv, html and pdf extesions are accepted'
        return extension
    def _load_file(self):
        # assert os.path.exists(self.file_name), f"{self.file_name} not found. Files available: {os.listdir()}"
        if self.file_extension.lower() == 'csv':
            self.loader = CSVLoader(file_path = self.file_path)
            self.documents = self.loader.load()
        elif self.file_extension.lower() == 'html':
            self.loader = UnstructuredHTMLLoader(file_path = self.file_path)
            self.documents = self.loader.load()
        else:
            self.loader = PyPDFLoader(file_path = self.file_path)
            self.documents = self.loader.load()
    def split_file(self, split_type = 'recursiv'):
        assert split_type in ['recursive', 'semantic'], "Please, be assured selecting a valid searchable engine: `recursive` and `semantic`."
        # Keeping the usage of RecursiveCharacterTextSplitter instead of CharacterTextSplitter
        # Considering that this way is feasible to use more separators
        if split_type == 'recursive':
            self.text_splitter = RecursiveCharacterTextSplitter(
                separators = self.separators,
                chunk_size = self.chunk_size,
                chunk_overlap = self.chunk_overlap
            )
            self.chunks = self.text_splitter.split_documents(self.documents)
        else:
        # Another way of segment chunks is by semantic meaning, expesivier but more certain what is being searched for here
            self.text_splitter = SemanticChunker(
                embeddings=AzureOpenAIEmbeddings(
                    openai_api_key=st.secrets['azure']['api_key'],
                    model="text-embedding-3-small",
                    azure_endpoint=st.secrets["azure"]["azure_endpoint"]
                ),
                breakpoint_threshold_type="gradient",
                breakpoint_threshold_amount=0.8
            )
            self.chunks = self.text_splitter.split_documents(self.documents)
    def _load_embedding_function(self):
        self.embedding_function = AzureOpenAIEmbeddings(
            openai_api_key=st.secrets['azure']['api_key'],
            model="text-embedding-3-small",
            azure_endpoint=st.secrets["azure"]["azure_endpoint"]
        )
    def _load_vector_store(self):
        persist_dir = tempfile.mkdtemp()
        self.vector_store = Chroma.from_documents(
            documents = self.chunks,
            embedding=self.embedding_function,
            # Doesn't need to persist the data in this case
            persist_directory=persist_dir
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k": 3}
        )
    def _load_language_model(self):
        self.llm = AzureChatOpenAI(
            api_key=st.secrets['azure']['api_key'],
            azure_endpoint=st.secrets["azure"]["azure_endpoint"],
            api_version=st.secrets["azure"]["api_version"],
            azure_deployment="gpt-4o-mini",
            temperature=0.0,
            max_tokens=4096,
            top_p=1.0,
        )
    def _load_template(self):
        self.prompt = ChatPromptTemplate.from_template(
            """
            Using the following pieces of context to answer the question at the end. If you don't know the answer, say you don't know.
            Questions may come in other languages, in this case reply into the anguage that it is asked.
            Context: {context}
            Question: {question}
            """
        )
    def _load_chain(self):
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    def _load_evaluators(self):
        self.faithfulness_chain = EvaluatorChain(
            metric = faithfulness,
            llm = self.llm,
            embeddings = self.embedding_function
        )
        
        self.context_chain = EvaluatorChain(
            metric = context_precision,
            llm = self.llm,
            embeddings = self.embedding_function
        )
    def invoke(self, question):
        response = self.chain.invoke(question)
        contexts = self.retriever.invoke(question)
        if self.file_extension == 'pdf':
            pages = [i.metadata.get('page') + 1 for i in contexts]
        elif self.file_extension == 'csv':
            pages = [i.metadata.get('row') + 1 for i in contexts]
        else:
            pages = ["No page reference for html files"]

        metric_faithfulness = self.faithfulness_chain.invoke({
          "question": question,
          "answer": response,
          "contexts": contexts
        }).get('faithfulness')

        metric_context = self.context_chain.invoke({
          "question": question,
          "ground_truth": response,
          "contexts": contexts
        }).get('context_precision')
        return {"response": response, "faithfulness": metric_faithfulness, "context_precision": metric_context, "pages": pages}

class PokemonAgent:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            api_key = st.secrets['azure']['api_key'],
            azure_endpoint = st.secrets["azure"]["azure_endpoint"],
            api_version = st.secrets["azure"]["api_version"],
            azure_deployment = "gpt-4o-mini",
            temperature=1.0,
            max_tokens=4096,
            top_p=1.0
        )
        self.model_with_tools = self.llm.bind_tools([self.identify_pokemon])
        self.POKEMON = None
        self.config = {"configurable": {"thread_id": "single_session_memory"}}
        self.app = self.creation_of_graph_workflow()

    @staticmethod
    @tool
    def identify_pokemon(description: str) -> str:
        """
        Attempts to identify a Pokémon based on the provided description.
    
        If the description is too vague or lacks sufficient detail for accurate identification,
        the tool will request additional information to proceed.
        
        Args:
            description (str): A textual description of the Pokémon (appearance, type, abilities, etc.).
    
        Returns:
            str: The name of the identified Pokémon or a prompt asking for more details.
        """
        llm = AzureChatOpenAI(
            api_key = st.secrets['azure']['api_key'],
            azure_endpoint = st.secrets["azure"]["azure_endpoint"],
            api_version = st.secrets["azure"]["api_version"],
            azure_deployment = "gpt-4o-mini",
            temperature=1.0,
            max_tokens=4096,
            top_p=1.0
        )
        
        messages = [
            SystemMessage(
                content=(
                    """
                    You are an AI Assistant and your function is to identified which is the pokemon that is based on the description given.
                    In case it is not possible to identify reply exactly the following between triple brackets: ```Not able to identified```, just reply its name, nothing more"""
                )
            ),
            HumanMessage(content=f"Discover which one is my pokemon based on the description. Description: {description}")
        ]
        response = llm.invoke(messages)
        return response.content

    def retrieve_last_human_message(self, state: MessagesState):
        """Function that retrieves the las human message"""
        return [msg for msg in state["messages"] if isinstance(msg, HumanMessage)][-1].content
    
    def retrieve_pokemon_ai_message(self, state: MessagesState):
        return [msg for msg in state["messages"] if isinstance(msg, AIMessage)][-1].content
    
    def call_initial_discover_pokemon(self, state: MessagesState):
        SYSTEM_PROMPT = """
        You are an AI Assistant and your function is to identified which is the pokemon that is based on the description given.
        In case it is not possible to identify which pokemon exactly, or you are not a 100 % sure, reply exactly the following between triple brackets: ```Not able to identified```.
        When is sure of the pokemon, just reply its name, nothing more
        """
        last_message = state["messages"]
    
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + last_message

        if self.POKEMON is not None:
            return {"messages": [AIMessage(content=f"Pokemon already descovered here: {self.POKEMON}. Reset the page if it's wanted another search.")]}
        elif isinstance(last_message, AIMessage) and last_message.tool_calls:
            return {"messages": [AIMessage(content=last_message.tool_calls[0]["response"])]}
        else:
            return {"messages": [self.model_with_tools.invoke(messages)]}
    
    def confirm_keep_discover_pokemon(self, state: MessagesState):
        last_message = state["messages"][-1]
    
        if (isinstance(last_message, ToolMessage)) and (last_message.content == "```Not able to identified```"):
            return {"messages": [AIMessage(content="Please, provide more info to identify the pokemon")]}
        elif self.retrieve_last_human_message(state).strip().lower() in ["yes", "y"]:
            self.POKEMON = self.retrieve_pokemon_ai_message(state)
            return {"messages": [AIMessage(content=f"Great! We discovered your Pokémon! {self.POKEMON}")]}
        else:
            return {"messages": [AIMessage(content=f"Is your pokemon {last_message.content}? Reply with **yes** if it is, or help giving more info about it :).")]}
    
    
    def should_continue(self, state: MessagesState):
        if self.POKEMON is not None:
            return END
        else:
            return "tools"

    def creation_of_graph_workflow(self):

        tool_node = ToolNode(tools = [self.identify_pokemon])
        workflow = StateGraph(MessagesState)
    
        workflow.add_node("chatbot", self.call_initial_discover_pokemon)
        workflow.add_node("evaluator", self.confirm_keep_discover_pokemon)
        workflow.add_node("tools", tool_node)
        
        workflow.add_edge(START, "chatbot")
        workflow.add_conditional_edges("chatbot", self.should_continue, [END, "tools"])
        workflow.add_edge("tools", "evaluator")
        workflow.add_edge("evaluator", END)
        
        memory = MemorySaver()
        app = workflow.compile(checkpointer = memory)

        return app

    def stream_memory_responses(self, user_input: str):
        print([event for event in self.app.stream({"messages": [("user", user_input)]}, self.config)])
        last_message = [event for event in self.app.stream({"messages": [("user", user_input)]}, self.config)][-1]
        last_key = list(last_message.keys())[0]
        return last_message.get(last_key).get('messages')[-1].content
        
