# Chatbot With Mongo + Mistral
- Streamlit: Creates the interface where Mistral API will be called;
- Mistral: `mistral-small-latest` is being used as default;
- Mongo: Every session that it is created has your history recorded there.

# Search Documents PDF
- Streamlit: Creates the interface where ChromaDB Retriever and Azure API will be called;
- Azure API: `gpt-4o-mini` is being used as default;
- ChromaDB: Every file uploaded is stored in order to act as a retriever.

# RAG LangChain LCEL
- Streamlit: Creates the interface where ChromaDB Retriever and Azure API will be called;
- Azure API: `gpt-4o-mini` is being used as default;
- ChromaDB (Local): Uploaded files are temporarily cached and indexed for retrieval; they do not persist after the session;
- Files Acepted: CSV, PDF and HTML.
- Architecture: RAG with Langchain using a LCEL architecture
  ```python
      self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )```