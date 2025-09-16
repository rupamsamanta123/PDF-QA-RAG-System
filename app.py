from langchain_community.document_loaders import PyPDFLoader
import os
import streamlit as st
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return ' '.join([doc.page_content for doc in documents])

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents([text])

# Configure embeddings for CPU usage
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def create_vector_store(chunks):
    return Chroma.from_documents(chunks, embeddings)

def generate_response(vector_store, query):
    llm = OllamaLLM(model="llama3.1")
    
    # Define a prompt template for the stuff documents chain
    prompt = ChatPromptTemplate.from_template(
        """Answer the following question based on the provided context:
        <context>
        {context}
        </context>
        Question: {input}"""
    )
    
    # Create the stuff documents chain
    chain = create_stuff_documents_chain(llm, prompt)
    
    # Perform similarity search to get relevant documents
    matching_docs = vector_store.similarity_search(query)
    
    # Invoke the chain with documents and query
    response = chain.invoke({"input": query, "context": matching_docs})
    
    # Return only the answer text
    return response

st.title("PDF Question-Answering App")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    st.write("Processing uploaded PDF...")
    raw_text = extract_text_from_pdf(temp_file_path)
    
    if raw_text:
        st.write("Splitting text into chunks...")
        chunks = split_text_into_chunks(raw_text)
        
        st.write("Creating vector store...")
        vector_store = create_vector_store(chunks)
        
        # User input for query
        query = st.text_input("Enter your query:")
        
        if query:
            st.write("Processing your query...")
            answer = generate_response(vector_store, query)
            st.write("Answer:", answer)
    
    # Clean up temporary file
    os.unlink(temp_file_path)
else:
    st.warning("Please upload a PDF file to proceed.")