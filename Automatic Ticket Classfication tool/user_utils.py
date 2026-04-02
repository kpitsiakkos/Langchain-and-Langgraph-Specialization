from pinecone import Pinecone as PineconeClient
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.openai_info import get_openai_callback
import joblib


def pull_from_pinecone(pinecone_apikey, pinecone_environment, pinecone_index_name, embeddings):
    pc = PineconeClient(api_key=pinecone_apikey)
    index = PineconeVectorStore.from_existing_index(
        index_name=pinecone_index_name,
        embedding=embeddings
    )
    return index


def create_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


def get_similar_docs(index, query, k=2):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs


def get_answer(docs, user_input):
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_input)
    return response


def predict(query_result):
    model = joblib.load('modelsvm.pk1')
    result = model.predict([query_result])
    return result[0]