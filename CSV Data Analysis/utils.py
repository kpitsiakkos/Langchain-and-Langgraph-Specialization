from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain_openai import OpenAI


def query_agent(data, query):

    # Parse the CSV file and create a Pandas DataFrame from its contents.
    # Gradio passes a file object, so we use .name to get the temp file path
    df = pd.read_csv(data.name)

    llm = OpenAI()

    # Create a Pandas DataFrame agent.
    agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

    # Invoke the agent and return only the output text
    response = agent.invoke(query)
    return response.get("output", response)