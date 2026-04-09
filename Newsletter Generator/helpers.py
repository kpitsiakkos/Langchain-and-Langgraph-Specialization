import os
import json
from dotenv import find_dotenv, load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_text_splitters import CharacterTextSplitter

load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings()


# 1. Serp request to get list of relevant articles
def search_serp(query):
    search = GoogleSerperAPIWrapper(k=5, type="search")
    response_json = search.results(query)
    print(f"Response=====>, {response_json}")
    return response_json


# 2. LLM to choose the best articles and return urls
def pick_best_articles_urls(response_json, query):
    response_str = json.dumps(response_json)

    llm = ChatOpenAI(temperature=0.7)
    template = """
      You are a world class journalist, researcher, tech, Software Engineer, Developer and a online course creator,
      you are amazing at finding the most interesting and relevant, useful articles in certain topics.

      QUERY RESPONSE:{response_str}

      Above is the list of search results for the query {query}.

      Please choose the best 3 articles from the list and return ONLY an array of the urls.
      Do not include anything else - return ONLY an array of the urls.
      Also make sure the articles are recent and not too old.
      If the file, or URL is invalid, show www.google.com.
    """
    prompt_template = PromptTemplate(
        input_variables=["response_str", "query"],
        template=template
    )

    # LCEL: prompt | llm  (replaces LLMChain)
    chain = prompt_template | llm

    result = chain.invoke({"response_str": response_str, "query": query})
    urls = json.loads(result.content)
    return urls


# 3. Load article content from urls and store in FAISS
def extract_content_from_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(data)
    db = FAISS.from_documents(docs, embeddings)
    return db


# 4. Summarise the articles
def summarizer(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    template = """
       {docs}
        As a world class journalist, researcher, article, newsletter and blog writer,
        you will summarize the text above in order to create a
        newsletter around {query}.
        This newsletter will be sent as an email. The format is going to be like
        Tim Ferriss' "5-Bullet Friday" newsletter.

        Please follow all of the following guidelines:
        1/ Make sure the content is engaging, informative with good data
        2/ Make sure the content is not too long, it should be the size of a nice newsletter bullet point and summary
        3/ The content should address the {query} topic very well
        4/ The content needs to be good and informative
        5/ The content needs to be written in a way that is easy to read, digest and understand
        6/ The content needs to give the audience actionable advice & insights including resources and links if necessary

        SUMMARY:
    """
    prompt_template = PromptTemplate(
        input_variables=["docs", "query"],
        template=template
    )

    chain = prompt_template | llm
    result = chain.invoke({"docs": docs_page_content, "query": query})
    return result.content.replace("\n", "")


# 5. Turn summarisation into a full newsletter
def generate_newsletter(summaries, query):
    summaries_str = str(summaries)

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    template = """
    {summaries_str}
        As a world class journalist, researcher, article, newsletter and blog writer,
        you'll use the text above as the context about {query}
        to write an excellent newsletter to be sent to subscribers about {query}.

        This newsletter will be sent as an email. The format is going to be like
        Tim Ferriss' "5-Bullet Friday" newsletter.

        Make sure to write it informally - no "Dear" or any other formalities. Start the newsletter with
        `Hi All!
          Here is your weekly dose of the Tech Newsletter, a list of what I find interesting
          and worth exploring.`

        Make sure to also write a backstory about the topic - make it personal, engaging and lighthearted before
        going into the meat of the newsletter.

        Please follow all of the following guidelines:
        1/ Make sure the content is engaging, informative with good data
        2/ Make sure the content is not too long, it should be the size of a nice newsletter bullet point and summary
        3/ The content should address the {query} topic very well
        4/ The content needs to be good and informative
        5/ The content needs to be written in a way that is easy to read, digest and understand
        6/ The content needs to give the audience actionable advice & insights including resources and links if necessary.

        If there are books or products involved, make sure to add amazon links to the products or just a link placeholder.

        As a signoff, write a clever quote related to learning, general wisdom, living a good life. Be creative with
        this one - and then sign with "Paulo
          - Learner and Teacher"

        NEWSLETTER-->:
    """
    prompt_template = PromptTemplate(
        input_variables=["summaries_str", "query"],
        template=template
    )

    chain = prompt_template | llm
    result = chain.invoke({"summaries_str": summaries_str, "query": query})
    return result.content