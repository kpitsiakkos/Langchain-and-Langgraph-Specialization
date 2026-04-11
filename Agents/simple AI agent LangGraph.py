import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ── 1. SETUP ──────────────────────────────────────────────────
load_dotenv()                                      # load .env file
openai_key = os.getenv("OPENAI_API_KEY")           # grab API key
llm_name = "gpt-3.5-turbo"                        # model to use
client = OpenAI(api_key=openai_key)                # raw OpenAI client (not used here but available)
model = ChatOpenAI(api_key=openai_key, model=llm_name)  # LangChain chat model


# ── 2. STATE ──────────────────────────────────────────────────
class State(TypedDict):
    # The graph passes this state between nodes.
    # add_messages APPENDS new messages instead of overwriting the list.
    messages: Annotated[list, add_messages]


# ── 3. BOT NODE ───────────────────────────────────────────────
def bot(state: State):
    # Send the full message history to the LLM and return its reply.
    # LangGraph automatically merges the returned messages into state.
    return {"messages": [model.invoke(state["messages"])]}


# ── 4. BUILD GRAPH ────────────────────────────────────────────
graph_builder = StateGraph(State)         # create a graph that uses our State
graph_builder.add_node("bot", bot)        # register the bot function as a node
graph_builder.set_entry_point("bot")      # first node to run when graph is invoked
graph_builder.set_finish_point("bot")     # last node — graph ends after bot replies
graph = graph_builder.compile()           # compile into a runnable graph


# ── 5. CHAT LOOP ──────────────────────────────────────────────
while True:
    user_input = input("User: ")

    if user_input.lower() in ["quit", "exit", "q"]:  # exit condition
        print("Goodbye!")
        break

    # stream() runs the graph and yields events as they are produced
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            # print only the last message (the bot's reply)
            print("Assistant:", value["messages"][-1].content)