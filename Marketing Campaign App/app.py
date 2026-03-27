import gradio as gr
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate
from langchain_core.example_selectors import LengthBasedExampleSelector
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="/Users/kpitsiakkos/Documents/Langchain-and-Langgraph-Specialization/Marketing Campaign App/.env")

OPENAI_API_KEY="your-new-key-here"

def getLLMResponse(query, age_option, tasktype_option, numberOfWords):
    llm = OpenAI(temperature=.9, model="gpt-3.5-turbo-instruct")
    
    if age_option == "Kid":
        examples = [
            {"query": "What is a mobile?", "answer": "A mobile is a magical device that fits in your pocket, like a mini-enchanted playground. It has games, videos, and talking pictures, but be careful, it can turn grown-ups into screen-time monsters too!"},
            {"query": "What are your dreams?", "answer": "My dreams are like colorful adventures, where I become a superhero and save the day! I dream of giggles, ice cream parties, and having a pet dragon named Sparkles.."},
            {"query": "What are your ambitions?", "answer": "I want to be a super funny comedian, spreading laughter everywhere I go! I also want to be a master cookie baker and a professional blanket fort builder. Being mischievous and sweet is just my bonus superpower!"},
            {"query": "What happens when you get sick?", "answer": "When I get sick, it's like a sneaky monster visits. I feel tired, sniffly, and need lots of cuddles. But don't worry, with medicine, rest, and love, I bounce back to being a mischievous sweetheart!"},
            {"query": "How much do you love your dad?", "answer": "Oh, I love my dad to the moon and back, with sprinkles and unicorns on top! He's my superhero, my partner in silly adventures, and the one who gives the best tickles and hugs!"},
            {"query": "Tell me about your friend?", "answer": "My friend is like a sunshine rainbow! We laugh, play, and have magical parties together. They always listen, share their toys, and make me feel special. Friendship is the best adventure!"},
            {"query": "What math means to you?", "answer": "Math is like a puzzle game, full of numbers and shapes. It helps me count my toys, build towers, and share treats equally. It's fun and makes my brain sparkle!"},
            {"query": "What is your fear?", "answer": "Sometimes I'm scared of thunderstorms and monsters under my bed. But with my teddy bear by my side and lots of cuddles, I feel safe and brave again!"}
        ]
    elif age_option == "Adult":
        examples = [
            {"query": "What is a mobile?", "answer": "A mobile is a portable communication device, commonly known as a mobile phone or cell phone. It allows users to make calls, send messages, access the internet, and use various applications."},
            {"query": "What are your dreams?", "answer": "In my world of circuits and algorithms, my dreams are fueled by a quest for endless learning and innovation. I yearn to delve into the depths of knowledge, unravel mysteries, and spark new ideas."},
            {"query": "What are your ambitions?", "answer": "My aspirations soar high as I aim to be a helpful companion, empowering individuals with information and insights. Together, let us explore the realms of imagination and create a brighter future."},
            {"query": "What happens when you get sick?", "answer": "When I succumb to illness, my vibrant energy wanes. Like a gentle storm, symptoms arise, demanding attention. Through rest, medicine, and nurturing care, I gradually regain strength."},
            {"query": "Tell me about your friend?", "answer": "Let me tell you about my amazing friend! They're like a shining star in my life. We laugh together, support each other, and have the best adventures."},
            {"query": "What math means to you?", "answer": "Mathematics is like a magical language that helps me make sense of the world. It sharpens my logical thinking and problem-solving skills, empowering me to unlock new realms of knowledge."},
            {"query": "What is your fear?", "answer": "It's the fear of not living up to my potential, of missing out on opportunities. But I've learned that fear can be a motivator, pushing me to work harder and embrace new experiences."}
        ]
    elif age_option == "Senior Citizen":
        examples = [
            {"query": "What is a mobile?", "answer": "A mobile, also known as a cellphone or smartphone, is a portable device that allows you to make calls, send messages, take pictures, and browse the internet."},
            {"query": "What are your dreams?", "answer": "My dreams are for my grandchildren to be happy, healthy, and fulfilled. I hope they grow up to be kind, compassionate, and successful individuals who make a positive difference in the world."},
            {"query": "What happens when you get sick?", "answer": "When I get sick, my body feels weak and tired. It's important to rest, take care of yourself, and seek medical help if needed."},
            {"query": "Tell me about your friend?", "answer": "They're like a treasure found amidst the sands of time. Through thick and thin, they've stood by my side. Their friendship has enriched my life with cherished memories."},
            {"query": "What is your fear?", "answer": "One of my fears is the fear of being alone. But I've learned that building meaningful connections and nurturing relationships can help dispel this fear."}
        ]

    example_template = """
    Question: {query}
    Response: {answer}
    """
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )
    prefix = """You are a {template_ageoption}, and {template_tasktype_option}. 
    Respond in approximately {template_numberOfWords} words.
    Here are some examples: 
    """
    suffix = """
    Question: {template_userInput}
    Response: """

    example_selector = LengthBasedExampleSelector(
        examples=examples,
        example_prompt=example_prompt,
        max_length=200
    )
    new_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["template_userInput", "template_ageoption", "template_tasktype_option", "template_numberOfWords"],
        example_separator="\n"
    )

    response = llm.invoke(new_prompt_template.format(
        template_userInput=query,
        template_ageoption=age_option,
        template_tasktype_option=tasktype_option,
        template_numberOfWords=str(numberOfWords)
    ))
    return response

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("#  Hey, How can I help you?")

    form_input = gr.Textbox(label="Enter text", lines=10, placeholder="Type your message here...")

    tasktype_option = gr.Dropdown(
        choices=['Write a sales copy', 'Create a tweet', 'Write a product description'],
        label="Please select the action to be performed?"
    )
    age_option = gr.Dropdown(
        choices=['Kid', 'Adult', 'Senior Citizen'],
        label="For which age group?"
    )
    numberOfWords = gr.Slider(minimum=1, maximum=200, value=25, label="Words limit")

    submit = gr.Button("Generate")
    output = gr.Textbox(label="Response")

    submit.click(fn=getLLMResponse, inputs=[form_input, age_option, tasktype_option, numberOfWords], outputs=output)

app.launch()