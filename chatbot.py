__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3
import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Initialize chat history
if 'responses' not in st.session_state:
    st.session_state['responses'] = []

# Streamlit app title
st.title("PDF Question Answering with Local LLM and RAG")

# Let the user upload multiple PDF files
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

# Initialize embeddings and vector store outside the loop
embeddings_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = None
# Query input: Chat-style
question = st.chat_input("Ask a question about the documents")
if uploaded_files:
    all_docs = []
    for uploaded_file in uploaded_files:
        # Load each PDF using PyPDFLoader
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())
        loader = PyPDFLoader(uploaded_file.name)
        pages = loader.load_and_split()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)
        all_docs.extend(docs)

    # Create or update the vector store
    if vector_store is None:
        vector_store = Chroma.from_documents(all_docs, embeddings_model)
    else:
        vector_store.add_documents(all_docs)

    # Set up retriever and LLM
    retriever = vector_store.as_retriever()
    llm = ChatOllama(model="llama3:latest", verbose=True)

    # Prompt template
    template = """Analyze the following context and answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)

    # Create the RAG chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    if question:
        # Process user question through RAG chain
        answer = chain.invoke(question)
        
        # Capitalize and refine answer for display
        answer = answer.capitalize()

        # Append user question and bot answer to session state
        st.session_state['responses'].append(("user", question))
        st.session_state['responses'].append(("bot", answer))

    # Display chat history
    st.subheader("Chat History")
    for role, message in st.session_state['responses']:
        q = 1
        if role == 'user':
            st.write(f"**You:Q-{q}** {message}")
        else:
            st.write(f"**Bot:Ans-{q}** {message}")
        q+=1

    # Option to delete the document collection from the vector store
    if st.button("Clear Document Collection"):
        vector_store.delete_collection()
        st.write("Document collection cleared.")
