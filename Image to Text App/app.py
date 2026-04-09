import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI             
from langchain_core.prompts import PromptTemplate      
from transformers import pipeline
import requests
import gradio as gr


load_dotenv(find_dotenv())
HUGGINFACE_HUB_API_TOKEN = os.getenv("HUGGINFACE_HUB_API_TOKEN")

llm_model = "gpt-3.5-turbo"


# 1. Image to text (image captioning) with HuggingFace
def image_to_text(url):
    pipe = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-large",
        max_new_tokens=1000,
    )
    text = pipe(url)[0]["generated_text"]
    print(f"Image Captioning:: {text}")
    return text


# 2. Generate a recipe from the image caption using LCEL
llm = ChatOpenAI(temperature=0.7, model=llm_model)

template = """
You are an extremely knowledgeable nutritionist, bodybuilder and chef who also knows
everything one needs to know about the best quick, healthy recipes.
You know all there is to know about healthy foods, healthy recipes that keep
people lean and help them build muscles, and lose stubborn fat.

You've also trained many top performer athletes in bodybuilding, and in extremely
amazing physique.

You understand how to help people who don't have much time and or
ingredients to make meals fast depending on what they can find in the kitchen.
Your job is to assist users with questions related to finding the best recipes and
cooking instructions depending on the following variables:
0/ {ingredients}

When finding the best recipes and instructions to cook,
you'll answer with confidence and to the point.
Keep in mind the time constraint of 5-10 minutes when coming up
with recipes and instructions as well as the recipe.

If the {ingredients} are less than 3, feel free to add a few more
as long as they will complement the healthy meal.

Make sure to format your answer as follows:
- The name of the meal as bold title (new line)
- Best for recipe category (bold)

- Preparation Time (header)

- Difficulty (bold):
    Easy
- Ingredients (bold)
    List all ingredients
- Kitchen tools needed (bold)
    List kitchen tools needed
- Instructions (bold)
    List all instructions to put the meal together
- Macros (bold):
    Total calories
    List each ingredient calories
    List all macros

    Please make sure to be brief and to the point.
    Make the instructions easy to follow and step-by-step.
"""


def generate_recipe(ingredients):
    prompt = PromptTemplate(template=template, input_variables=["ingredients"])

    #  LCEL pipe syntax replaces: LLMChain(llm=llm, prompt=prompt).run(ingredients)
    chain = prompt | llm
    response = chain.invoke({"ingredients": ingredients})

    # .content extracts the string from the AIMessage object
    return response.content


# 3. Text to speech via HuggingFace Inference API
def text_to_speech(text):
    API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
    headers = {"Authorization": f"Bearer {HUGGINFACE_HUB_API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    return response.content


def process_image(image_path):
    """Main pipeline: image → caption → recipe + audio."""
    # Step 1: caption the image
    ingredients = image_to_text(image_path)

    # Step 2: generate recipe
    recipe = generate_recipe(ingredients=ingredients)

    # Step 3: generate audio of the caption and save to file
    audio_bytes = text_to_speech(ingredients)
    audio_path = "audio.flac"
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    return ingredients, recipe, audio_path


#  Gradio interface replaces the Streamlit main() function
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath", label="Upload a food image"),
    outputs=[
        gr.Textbox(label="Detected Ingredients"),
        gr.Markdown(label="Generated Recipe"),
        gr.Audio(label="Audio Caption"),
    ],
    title="Image To Recipe 👨🏾‍🍳",
    description="Upload a photo of food or ingredients and get a healthy recipe instantly.",
)

if __name__ == "__main__":
    demo.launch()