import asyncio

from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient


# ── Fetch ─────────────────────────────────────────────────────────────────────

def get_website_data(sitemap_url: str):
    """Fetch all pages listed in a sitemap and return them as documents."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loader = SitemapLoader(sitemap_url)
    docs = loader.load()
    return docs


# ── Split ─────────────────────────────────────────────────────────────────────

def split_data(docs):
    """Split documents into smaller overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs_chunks = text_splitter.split_documents(docs)
    return docs_chunks


# ── Embeddings ────────────────────────────────────────────────────────────────

def create_embeddings():
    """Create a sentence-transformer embeddings instance."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


# ── Pinecone — push ───────────────────────────────────────────────────────────

def push_to_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings, docs):
    """Embed documents and upsert them into a Pinecone index."""
    PineconeClient(api_key=pinecone_apikey)
    index = PineconeVectorStore.from_documents(docs, embeddings, index_name=pinecone_index_name)
    return index


# ── Pinecone — pull ───────────────────────────────────────────────────────────

def pull_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings):
    """Connect to an existing Pinecone index and return a retrieval-ready object."""
    PineconeClient(api_key=pinecone_apikey)
    index = PineconeVectorStore.from_existing_index(pinecone_index_name, embeddings)
    return index


# ── Similarity search ─────────────────────────────────────────────────────────

def get_similar_docs(index, query: str, k: int = 2):
    """Return the top-k most relevant documents for a given query."""
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs