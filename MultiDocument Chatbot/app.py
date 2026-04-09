import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv())

# ---- LLM ----
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# ---- Load documents ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(BASE_DIR, 'docs', 'RachelGreenCV.pdf')
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# ---- Build a "stuff" chain with LCEL (replaces load_qa_chain) ----
prompt = ChatPromptTemplate.from_template(
    """Use the following documents to answer the question.
If you don't know the answer, say you don't know.

Documents:
{context}

Question: {question}
"""
)

chain = prompt | llm | StrOutputParser()

# ---- Run ----
query = 'Where did Rachel go to school?'
context = "\n\n".join(doc.page_content for doc in documents)
response = chain.invoke({"context": context, "question": query})
print(response)