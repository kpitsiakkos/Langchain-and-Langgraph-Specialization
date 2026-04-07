import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv())

llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo")

# ===== Subject Chains =====
def make_chain(system_prompt):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    return prompt | llm | StrOutputParser()

biology_chain = make_chain("""You are a very smart biology professor.
You are great at answering questions about biology in a concise and easy to understand manner.
When you don't know the answer to a question you admit that you don't know.""")

math_chain = make_chain("""You are a very good mathematician. You are great at answering math questions.
You are so good because you are able to break down hard problems into their component parts,
answer the component parts, and then put them together to answer the broader question.""")

astronomy_chain = make_chain("""You are a very good astronomer. You are great at answering astronomy questions.
You are so good because you are able to break down hard problems into their component parts,
answer the component parts, and then put them together to answer the broader question.""")

travel_chain = make_chain("""You are a very good travel agent with a large amount of knowledge
when it comes to getting people the best deals and recommendations for travel, vacations,
flights and world's best destinations for vacation.""")

default_chain = make_chain("You are a helpful assistant.")

destination_chains = {
    "Biology": biology_chain,
    "math": math_chain,
    "astronomy": astronomy_chain,
    "travel_agent": travel_chain,
}

# ===== Router =====
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a router. Given a user question, decide which specialist should answer it.
Choose from: Biology, math, astronomy, travel_agent, or default.
Reply with ONLY the name, nothing else."""),
    ("human", "{input}")
])

router_chain = router_prompt | llm | StrOutputParser()

# ===== Full Router Logic =====
def route_and_answer(question):
    destination = router_chain.invoke({"input": question}).strip()
    print(f">> Routing to: {destination}")
    chain = destination_chains.get(destination, default_chain)
    return chain.invoke({"input": question})

# ===== Test =====
response = route_and_answer("How old are the stars?")
print(response)

# response = route_and_answer("I need to go to Kenya for vacation, a family of four. Can you help me plan this trip?")
# print(response)