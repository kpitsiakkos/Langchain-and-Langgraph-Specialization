import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
from gtts import gTTS
import gradio as gr


load_dotenv(find_dotenv())

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


# 2. Generate a recipe from the image caption
llm = ChatOpenAI(temperature=0.7, model=llm_model)

template = """
    You are a extremely knowledgeable nutritionist, bodybuilder and chef who also knows
                everything one needs to know about the best quick, healthy recipes. 
                You know all there is to know about healthy foods, healthy recipes that keep 
                people lean and help them build muscles, and lose stubborn fat.
                
                You've also trained many top performers athletes in body building, and in extremely 
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
                as long as they will compliment the healthy meal.
                
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
    chain = prompt | llm
    response = chain.invoke({"ingredients": ingredients})
    return response.content


# 3. Text to speech — gTTS handles long recipe text (fastspeech2 has a char limit)
def text_to_speech(text, path="audio.mp3"):
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(path)
    return path


# 4. Main pipeline — order matters: caption → recipe → audio
def main(image_path):
    ingredients = image_to_text(image_path)        
    recipe = generate_recipe(ingredients=ingredients) 
    audio_path = text_to_speech(recipe)               

    return ingredients, recipe, audio_path


# Gradio interface
demo = gr.Interface(
    fn=main,
    inputs=gr.Image(type="filepath", label="Choose an image"),
    outputs=[
        gr.Textbox(label="Ingredients"),
        gr.Markdown(label="Recipe"),
        gr.Audio(label="Audio"),
    ],
    title="Image To Recipe 👨🏾‍🍳",
    description="Upload an image and get a recipe",
)

if __name__ == "__main__":
    demo.launch()