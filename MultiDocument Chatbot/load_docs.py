from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, 'docs')

def load_docs():
    documents = []
    for file in os.listdir(DOCS_DIR):
        if file.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(DOCS_DIR, file))
            documents.extend(loader.load())
        elif file.endswith('.docx') or file.endswith('.doc'):
            loader = Docx2txtLoader(os.path.join(DOCS_DIR, file))
            documents.extend(loader.load())
        elif file.endswith('.txt'):
            loader = TextLoader(os.path.join(DOCS_DIR, file))
            documents.extend(loader.load())

    return documents