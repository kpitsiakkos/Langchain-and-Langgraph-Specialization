import os
from dotenv import find_dotenv, load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import gradio as gr

load_dotenv(find_dotenv())

open_ai = OpenAI(temperature=0.7, model="gpt-3.5-turbo-instruct")

def generate_lullaby(location, name, language):
    if not location or not name or not language:
        return "Please fill in all fields.", ""

    # Chain 1: Generate lullaby
    prompt_story = PromptTemplate(
        input_variables=["location", "name"],
        template="""As a children's book writer, please come up with a simple and short (90 words)
        lullaby based on the location {location} and the main character {name}. STORY:"""
    )

    # Chain 2: Translate
    prompt_translate = PromptTemplate(
        input_variables=["story", "language"],
        template="""Translate the {story} into {language}. Make sure the language is simple and fun. TRANSLATION:"""
    )

    # Build chains using LCEL
    story_chain = prompt_story | open_ai
    translate_chain = prompt_translate | open_ai

    # Run sequentially
    story = story_chain.invoke({"location": location, "name": name})
    translated = translate_chain.invoke({"story": story, "language": language})

    return story, translated


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📖 Let AI Write and Translate a Lullaby for You")

    with gr.Row():
        location_input = gr.Textbox(label="Where is the story set?", placeholder="e.g. Zanzibar")
        name_input = gr.Textbox(label="Main character's name", placeholder="e.g. Maya")
        language_input = gr.Textbox(label="Translate into...", placeholder="e.g. French")

    submit_btn = gr.Button("✨ Generate Lullaby", variant="primary")

    with gr.Row():
        english_output = gr.Textbox(label="🇬🇧 English Version", lines=8)
        translated_output = gr.Textbox(label="🌍 Translated Version", lines=8)

    status = gr.Markdown("")

    def run(location, name, language):
        story, translated = generate_lullaby(location, name, language)
        return story, translated, "✅ Lullaby successfully generated!"

    submit_btn.click(
        fn=run,
        inputs=[location_input, name_input, language_input],
        outputs=[english_output, translated_output, status],
    )

demo.launch()