from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_community.llms import HuggingFaceHub
from langchain_classic.chains.summarize import load_summarize_chain
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
import os
import uuid
import time


# Extract Information from PDF file
def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Iterate over files that user uploaded PDF files, one by one
def create_docs(user_pdf_list, unique_id):
    docs = []
    for filepath in user_pdf_list:
        chunks = get_pdf_text(filepath)
        docs.append(Document(
            page_content=chunks,
            metadata={
                "name": os.path.basename(filepath),
                "id": uuid.uuid4().hex,
                "type": "application/pdf",
                "size": os.path.getsize(filepath),
                "unique_id": unique_id
            },
        ))
    return docs


# Create embeddings instance
def create_embeddings_load_data():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


# Function to push data to Vector Store - Pinecone
def push_to_pinecone(pinecone_apikey, pinecone_index_name, embeddings, docs):
    os.environ["PINECONE_API_KEY"] = pinecone_apikey
    PineconeVectorStore.from_documents(docs, embeddings, index_name=pinecone_index_name)


# Function to pull information from Vector Store - Pinecone
def pull_from_pinecone(pinecone_apikey, pinecone_index_name, embeddings):
    print("20secs delay...")
    time.sleep(20)
    os.environ["PINECONE_API_KEY"] = pinecone_apikey
    index = PineconeVectorStore.from_existing_index(pinecone_index_name, embeddings)
    return index


# Function to get relevant documents from vector store based on user input
def similar_docs(query, k, pinecone_apikey, pinecone_index_name, embeddings, unique_id):
    index = pull_from_pinecone(pinecone_apikey, pinecone_index_name, embeddings)
    results = index.similarity_search_with_score(query, int(k), filter={"unique_id": unique_id})
    return results


# Helps us get the summary of a document
def get_summary(current_doc):
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run([current_doc])
    return summary