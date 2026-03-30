from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun


# Function to generate video script
def generate_script(prompt, video_length, creativity, api_key):

    # Template for generating 'Title'
    title_template = PromptTemplate(
        input_variables=['subject'],
        template='Please come up with a title for a YouTube video on the {subject}.'
    )

    # Template for generating 'Video Script' using search engine
    script_template = PromptTemplate(
        input_variables=['title', 'DuckDuckGo_Search', 'duration'],
        template='Create a script for a YouTube video based on this title for me. TITLE: {title} of duration: {duration} minutes using this search data {DuckDuckGo_Search} '
    )

    # Setting up OpenAI LLM
    llm = ChatOpenAI(temperature=creativity, openai_api_key=api_key,
                     model_name='gpt-3.5-turbo')

    # Creating chains using LCEL
    title_chain = title_template | llm | StrOutputParser()
    script_chain = script_template | llm | StrOutputParser()

    # https://python.langchain.com/docs/modules/agents/tools/integrations/ddg
    search = DuckDuckGoSearchRun()

    # Executing the chains we created for 'Title'
    title = title_chain.invoke({'subject': prompt})

    # Executing the chains we created for 'Video Script' by taking help of search engine 'DuckDuckGo'
    search_result = search.run(prompt)
    script = script_chain.invoke({'title': title, 'DuckDuckGo_Search': search_result, 'duration': video_length})

    # Returning the output
    return search_result, title, script