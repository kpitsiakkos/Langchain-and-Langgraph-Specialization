import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_tavily import TavilySearch              # updated package
from langgraph.graph.message import add_messages
import json
from langchain_core.messages import ToolMessage, BaseMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver    # built-in, no extra package needed

# ── 1. SETUP ──────────────────────────────────────────────────
load_dotenv()                                           # load .env file
openai_key = os.getenv("OPENAI_API_KEY")                # grab OpenAI key
tavily = os.getenv("TAVILY_API_KEY")                    # grab Tavily key
llm_name = "gpt-3.5-turbo"                             # model to use
client = OpenAI(api_key=openai_key)                     # raw OpenAI client
model = ChatOpenAI(api_key=openai_key, model=llm_name)  # LangChain chat model

# ── 2. STATE ──────────────────────────────────────────────────
class State(TypedDict):
    # The graph passes this state between nodes.
    # add_messages APPENDS new messages instead of overwriting the list.
    messages: Annotated[list, add_messages]

# ── 3. TOOLS ──────────────────────────────────────────────────
tool = TavilySearch(max_results=2)  # web search tool — returns top 2 results
tools = [tool]                      # list of all available tools

# bind_tools() tells the LLM which tools exist and how to call them
# the model can now decide ON ITS OWN when to use a tool
model_with_tools = model.bind_tools(tools)

# ── 4. BOT NODE ───────────────────────────────────────────────
def bot(state: State):
    # invoke the model with the full message history
    # if the model decides to use a tool, it returns a tool_call instead of plain text
    return {"messages": [model_with_tools.invoke(state["messages"])]}

# ── 5. TOOL NODE ──────────────────────────────────────────────
# ToolNode automatically runs whichever tool the model requested
# it reads the tool_call from the last message and returns the result
tool_node = ToolNode(tools=[tool])

# ── 6. BUILD GRAPH ────────────────────────────────────────────
graph_builder = StateGraph(State)           # create graph with our State
graph_builder.add_node("bot", bot)          # register bot as a node
graph_builder.add_node("tools", tool_node)  # register tool executor as a node
graph_builder.set_entry_point("bot")        # always start at bot

# tools_condition checks the last message:
#   → if it contains a tool_call  : route to "tools" node
#   → if it is a plain text reply : route to END
graph_builder.add_conditional_edges("bot", tools_condition)
# after tools run, always return to bot so it can read the result and reply
graph_builder.add_edge("tools", "bot")

# ── 7. MEMORY ─────────────────────────────────────────────────
# MemorySaver stores the graph state in RAM after every step
# no external database needed — data is lost when the script stops
memory = MemorySaver()

# compile the graph WITH the memory checkpointer attached
graph = graph_builder.compile(checkpointer=memory)

# ── 8. THREAD CONFIG ──────────────────────────────────────────
# thread_id groups messages into a single conversation session
# memory is saved and retrieved per thread_id
config = {"configurable": {"thread_id": 1}}

# ── 9. FIRST MESSAGE ──────────────────────────────────────────
user_input = "Hi there! My name is Bond. and I have been happy for 100 years"

# stream_mode="values" yields the full state after every node runs
events = graph.stream(
    {"messages": [("user", user_input)]},
    config,                # pass config so memory knows which thread to save to
    stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()  # print the latest message neatly

# ── 10. SECOND MESSAGE (memory test) ──────────────────────────
user_input = "do you remember my name, and how long have I been happy for?"

# the graph loads the previous messages from memory using thread_id
# so the bot already knows the name and the 100 years from turn 1
events = graph.stream(
    {"messages": [("user", user_input)]},
    config,
    stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()  # should recall Bond + 100 years

# ── 11. CONTINUE CHATTING ─────────────────────────────────────
# hand control to the user — memory is still active from the two tests above
# the bot already knows Bond and 100 years, so context carries over
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:  # exit condition
        print("Goodbye!")
        break
    events = graph.stream(
        {"messages": [("user", user_input)]},
        config,               # same thread_id = bot remembers everything
        stream_mode="values"
    )
    for event in events:
        event["messages"][-1].pretty_print()  # print the bot's reply