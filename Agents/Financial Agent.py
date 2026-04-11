import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
import json
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel                                # fixed import
import pandas as pd
from io import StringIO
import gradio as gr

# ── 1. SETUP ──────────────────────────────────────────────────
load_dotenv()                                                 # load .env file
openai_key = os.getenv("OPENAI_API_KEY")                      # grab OpenAI key
tavily_key = os.getenv("TAVILY_API_KEY")                      # grab Tavily key
llm_name = "gpt-3.5-turbo"                                   # model to use
model = ChatOpenAI(api_key=openai_key, model=llm_name)        # LangChain chat model
memory = MemorySaver()                                        # in-RAM memory checkpointer

from tavily import TavilyClient
tavily = TavilyClient(api_key=tavily_key)                     # Tavily client for web search

# ── 2. STATE ──────────────────────────────────────────────────
class AgentState(TypedDict):
    # shared state passed between every node in the graph
    task: str             # the user's analysis task
    competitors: List[str]# list of competitor company names
    csv_file: str         # raw CSV content uploaded by the user
    financial_data: str   # parsed financial data from CSV
    analysis: str         # LLM analysis of the financial data
    competitor_data: str  # raw competitor research data
    comparison: str       # comparison between company and competitors
    feedback: str         # reviewer feedback on the comparison
    report: str           # final written report
    content: List[str]    # web search results accumulated across nodes
    revision_number: int  # tracks how many revision cycles have run
    max_revisions: int    # maximum allowed revision cycles before ending

# ── 3. STRUCTURED OUTPUT ──────────────────────────────────────
class Queries(BaseModel):
    # forces the LLM to return a structured list of search queries
    queries: List[str]

# ── 4. PROMPTS ────────────────────────────────────────────────
GATHER_FINANCIALS_PROMPT    = """You are an expert financial analyst. Gather the financial data for the given company. Provide detailed financial data."""
ANALYZE_DATA_PROMPT         = """You are an expert financial analyst. Analyze the provided financial data and provide detailed insights and analysis."""
RESEARCH_COMPETITORS_PROMPT = """You are a researcher tasked with providing information about similar companies for performance comparison. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""
COMPETE_PERFORMANCE_PROMPT  = """You are an expert financial analyst. Compare the financial performance of the given company with its competitors based on the provided data.
**MAKE SURE TO INCLUDE THE NAMES OF THE COMPETITORS IN THE COMPARISON.**"""
FEEDBACK_PROMPT             = """You are a reviewer. Provide detailed feedback and critique for the provided financial comparison report. Include any additional information or revisions needed."""
WRITE_REPORT_PROMPT         = """You are a financial report writer. Write a comprehensive financial report based on the analysis, competitor research, comparison, and feedback provided."""
RESEARCH_CRITIQUE_PROMPT    = """You are a researcher tasked with providing information to address the provided critique. Generate a list of search queries to gather relevant information. Only generate 3 queries max."""

# ── 5. NODES ──────────────────────────────────────────────────
def gather_financials_node(state: AgentState):
    csv_file = state["csv_file"]                              # get raw CSV string from state
    df = pd.read_csv(StringIO(csv_file))                      # parse CSV into a DataFrame
    financial_data_str = df.to_string(index=False)            # convert DataFrame to plain text
    combined_content = f"{state['task']}\n\nHere is the financial data:\n\n{financial_data_str}"
    messages = [
        SystemMessage(content=GATHER_FINANCIALS_PROMPT),      # set the analyst role
        HumanMessage(content=combined_content),               # send task + financial data
    ]
    response = model.invoke(messages)                         # ask LLM to summarise financials
    return {"financial_data": response.content}               # save result to state


def analyze_data_node(state: AgentState):
    messages = [
        SystemMessage(content=ANALYZE_DATA_PROMPT),           # set the analyst role
        HumanMessage(content=state["financial_data"]),        # send the gathered financial data
    ]
    response = model.invoke(messages)                         # ask LLM for deep analysis
    return {"analysis": response.content}                     # save analysis to state


def research_competitors_node(state: AgentState):
    content = state.get("content") or []                      # use .get() to avoid KeyError on first run
    for competitor in state["competitors"]:                   # loop through each competitor
        queries = model.with_structured_output(Queries).invoke(
            [
                SystemMessage(content=RESEARCH_COMPETITORS_PROMPT),  # set researcher role
                HumanMessage(content=competitor),                    # ask for search queries
            ]
        )
        for q in queries.queries:                             # run each generated search query
            response = tavily.search(query=q, max_results=2) # search the web via Tavily
            for r in response["results"]:
                content.append(r["content"])                  # collect search result text
    return {"content": content}                               # save all results to state


def compare_performance_node(state: AgentState):
    content = "\n\n".join(state.get("content") or [])        # use .get() to avoid KeyError on first run
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is the financial analysis:\n\n{state['analysis']}"
    )
    messages = [
        SystemMessage(content=COMPETE_PERFORMANCE_PROMPT.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)                         # ask LLM to compare company vs competitors
    return {
        "comparison": response.content,                       # save comparison to state
        "revision_number": state.get("revision_number", 1) + 1,  # increment revision counter
    }


def collect_feedback_node(state: AgentState):
    messages = [
        SystemMessage(content=FEEDBACK_PROMPT),               # set reviewer role
        HumanMessage(content=state["comparison"]),            # send the current comparison
    ]
    response = model.invoke(messages)                         # ask LLM to critique the comparison
    return {"feedback": response.content}                     # save feedback to state


def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),  # set researcher role
            HumanMessage(content=state["feedback"]),          # send feedback to generate new queries
        ]
    )
    content = state.get("content") or []                      # use .get() to avoid KeyError on first run
    for q in queries.queries:                                 # run each query to address the critique
        response = tavily.search(query=q, max_results=2)      # search the web
        for r in response["results"]:
            content.append(r["content"])                      # add new results to content
    return {"content": content}                               # save updated content to state


def write_report_node(state: AgentState):
    messages = [
        SystemMessage(content=WRITE_REPORT_PROMPT),           # set report writer role
        HumanMessage(content=state["comparison"]),            # send the final comparison
    ]
    response = model.invoke(messages)                         # ask LLM to write the full report
    return {"report": response.content}                       # save report to state

# ── 6. CONDITIONAL EDGE ───────────────────────────────────────
def should_continue(state):
    # if max revisions reached → end the graph
    # otherwise → collect more feedback and revise
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "collect_feedback"

# ── 7. BUILD GRAPH ────────────────────────────────────────────
builder = StateGraph(AgentState)                              # create graph with AgentState

builder.add_node("gather_financials",    gather_financials_node)    # node 1: read CSV + summarise
builder.add_node("analyze_data",         analyze_data_node)         # node 2: deep analysis
builder.add_node("research_competitors", research_competitors_node) # node 3: web search competitors
builder.add_node("compare_performance",  compare_performance_node)  # node 4: compare vs competitors
builder.add_node("collect_feedback",     collect_feedback_node)     # node 5: critique the comparison
builder.add_node("research_critique",    research_critique_node)    # node 6: search to address critique
builder.add_node("write_report",         write_report_node)         # node 7: write the final report

builder.set_entry_point("gather_financials")                  # always start by reading the CSV

builder.add_conditional_edges(
    "compare_performance",
    should_continue,
    {END: END, "collect_feedback": "collect_feedback"},       # route to END or feedback loop
)

builder.add_edge("gather_financials",    "analyze_data")      # step 1 → 2
builder.add_edge("analyze_data",         "research_competitors") # step 2 → 3
builder.add_edge("research_competitors", "compare_performance")  # step 3 → 4
builder.add_edge("collect_feedback",     "research_critique")    # step 5 → 6
builder.add_edge("research_critique",    "compare_performance")  # step 6 → back to 4
builder.add_edge("compare_performance",  "write_report")         # step 4 → 7 (when done)

graph = builder.compile(checkpointer=memory)                  # compile graph with memory

# ── 8. GRADIO ANALYSIS FUNCTION ───────────────────────────────
def run_analysis(task, competitors_text, max_revisions, csv_file):
    if csv_file is None:                                       # guard: no file uploaded
        return "Please upload a CSV file.", ""

    with open(csv_file.name, "r") as f:                       # read the uploaded CSV
        csv_data = f.read()

    competitors = [c.strip() for c in competitors_text.split("\n") if c.strip()]  # clean list

    initial_state = {
        "task": task,
        "competitors": competitors,
        "csv_file": csv_data,
        "max_revisions": int(max_revisions),
        "revision_number": 1,                                 # start revision count at 1
    }
    thread = {"configurable": {"thread_id": "1"}}             # thread for memory tracking

    log = ""          # accumulate node-by-node output for the progress box
    final_report = "" # store the final report separately

    for s in graph.stream(initial_state, thread):             # stream graph events in real time
        log += str(s) + "\n\n"                                # append each node's output to log
        if "write_report" in s:
            final_report = s["write_report"]["report"]        # extract final report when ready

    return log, final_report                                   # return both to Gradio outputs

# ── 9. GRADIO UI ──────────────────────────────────────────────
with gr.Blocks(title="Financial Performance Reporting Agent") as demo:  # theme moved to launch()

    gr.Markdown("## Financial Performance Reporting Agent")
    gr.Markdown("Upload your company's financial CSV and compare against competitors.")

    with gr.Row():
        with gr.Column():
            task_input = gr.Textbox(
                label="Task",
                value="Analyze the financial performance of our company (MyAICo.AI) compared to competitors",
            )                                                  # user types the analysis task
            competitors_input = gr.Textbox(
                label="Competitors (one per line)",
                placeholder="Microsoft\nNvidia\nGoogle",
                lines=4,
            )                                                  # one competitor per line
            max_revisions_input = gr.Number(
                label="Max Revisions", value=2, minimum=1
            )                                                  # how many feedback loops to allow
            csv_upload = gr.File(
                label="Upload CSV with financial data", file_types=[".csv"]
            )                                                  # CSV upload widget
            run_btn = gr.Button("Start Analysis", variant="primary")  # trigger button

        with gr.Column():
            progress_output = gr.Textbox(
                label="Agent Progress (node by node)", lines=20
            )                                                  # shows each node's raw output
            report_output = gr.Textbox(
                label="Final Report", lines=20
            )                                                  # shows the final written report

    # on button click — run the full graph and populate both output boxes
    run_btn.click(
        run_analysis,
        inputs=[task_input, competitors_input, max_revisions_input, csv_upload],
        outputs=[progress_output, report_output],
    )

# ── 10. LAUNCH ────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())