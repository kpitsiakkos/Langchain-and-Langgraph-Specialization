import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from load_docs import load_docs
import gradio as gr

load_dotenv(find_dotenv())

# ---- LLM ----
llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Load & split documents ----
documents = load_docs()

text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=10)
docs = text_splitter.split_documents(documents)

# ---- Vector store ----
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory=os.path.join(BASE_DIR, 'data')
)
retriever = vectordb.as_retriever(search_kwargs={'k': 6})

# ---- Step 1: rephrase the question given chat history ----
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given the chat history and the latest user question, "
     "formulate a standalone question that can be understood without the history. "
     "Do NOT answer it — just rephrase if needed, otherwise return it as-is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
contextualize_chain = contextualize_prompt | llm | StrOutputParser()

def get_standalone_question(x):
    if x.get("chat_history"):
        return contextualize_chain.invoke(x)
    return x["input"]

# ---- Step 2: retrieve docs based on standalone question ----
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ---- Step 3: answer using retrieved context ----
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant for question-answering tasks. "
     "Use the retrieved context below to answer the question. "
     "If you don't know the answer, say so.\n\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

rag_chain = (
    RunnablePassthrough.assign(standalone_question=get_standalone_question)
    | RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["standalone_question"])))
    | qa_prompt
    | llm
    | StrOutputParser()
)


# ---- Gradio UI ----
# Gradio passes chat history as a list of dicts or tuples depending on version
def build_chat_history(gradio_history):
    history = []
    for msg in gradio_history:
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                history.append(HumanMessage(content=content))
            elif role == "assistant":
                history.append(AIMessage(content=content))
        else:
            # fallback: tuple (user_msg, bot_msg)
            user_msg, bot_msg = msg[0], msg[1]
            history.append(HumanMessage(content=user_msg))
            if bot_msg:
                history.append(AIMessage(content=bot_msg))
    return history


def chat(user_input, gradio_history):
    chat_history = build_chat_history(gradio_history)
    return rag_chain.invoke({"input": user_input, "chat_history": chat_history})


demo = gr.ChatInterface(
    fn=chat,
    title="Docs QA Bot using Langchain",
    chatbot=gr.Chatbot(placeholder="Ask anything about your documents... 🤖"),
    textbox=gr.Textbox(placeholder="Ask a question about your documents...", container=False),
)

if __name__ == "__main__":
    demo.launch()