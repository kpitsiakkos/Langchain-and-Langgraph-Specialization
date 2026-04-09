import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(find_dotenv())

# ---- LLM ----
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# ---- Load & split ----
loader = PyPDFLoader('./docs/RachelGreenCV.pdf')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# ---- Vector store (no need to call .persist() in newer chromadb) ----
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory='./data'
)

# ---- LCEL retrieval chain (replaces RetrievalQA) ----
retriever = vectordb.as_retriever(search_kwargs={'k': 3})

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
{context}

Question: {question}
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---- Run ----
result = chain.invoke("When did Rachel graduate?")
print(result)
