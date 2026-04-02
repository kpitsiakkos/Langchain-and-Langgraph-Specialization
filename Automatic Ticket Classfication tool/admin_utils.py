from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
import pandas as pd
from sklearn.model_selection import train_test_split


# ── PDF / Pinecone helpers ────────────────────────────────────────────────────

def read_pdf_data(pdf_path):
    """Read and extract all text from a PDF file."""
    pdf_page = PdfReader(pdf_path)
    text = ""
    for page in pdf_page.pages:
        text += page.extract_text()
    return text


def split_data(text):
    """Split text into overlapping chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_text(text)
    docs_chunks = text_splitter.create_documents(docs)
    return docs_chunks


def create_embeddings_load_data():
    """Create HuggingFace embeddings instance for PDF loading."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs):
    """Push document embeddings into a Pinecone index."""
    PineconeClient(api_key=pinecone_apikey)
    index = PineconeVectorStore.from_documents(docs, embeddings, index_name=pinecone_index_name)
    return index


# ── ML Model helpers ──────────────────────────────────────────────────────────

def read_data(csv_path):
    """Read the labelled CSV dataset. Expects two columns: text, label."""
    df = pd.read_csv(csv_path, delimiter=',', header=None)
    return df


def get_embeddings():
    """Create HuggingFace embeddings instance for ML model."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


def create_embeddings_for_df(df, embeddings):
    """Generate embeddings for every row in the dataframe."""
    df[2] = df[0].apply(lambda x: embeddings.embed_query(x))
    return df


def split_train_test__data(df_sample):
    """Split the dataset into training and testing sets (75/25)."""
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(
        list(df_sample[2]), list(df_sample[1]), test_size=0.25, random_state=0
    )
    return sentences_train, sentences_test, labels_train, labels_test


def get_score(svm_classifier, sentences_test, labels_test):
    """Return the accuracy score of the trained classifier on test data."""
    score = svm_classifier.score(sentences_test, labels_test)
    return score