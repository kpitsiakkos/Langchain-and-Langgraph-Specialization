import gradio as gr
from utils import generate_script


def generate(prompt, video_length, creativity, api_key):
    if not api_key:
        return "Ooopssss!!! Please provide API key.....", "", ""

    search_result, title, script = generate_script(prompt, video_length, creativity, api_key)
    return title, script, search_result


with gr.Blocks(title="YouTube Script Writing Tool") as demo:
    gr.Markdown("# ❤️ YouTube Script Writing Tool")

    with gr.Sidebar():
        gr.Markdown("## 😎🗝️")
        api_key = gr.Textbox(label="What's your API key?", type="password")
        gr.Image("./Youtube.jpg", show_label=False)

    # Captures User Inputs
    prompt = gr.Textbox(label="Please provide the topic of the video")
    video_length = gr.Textbox(label="Expected Video Length 🕒 (in minutes)")
    creativity = gr.Slider(label="Creativity limit ✨ - (0 LOW || 1 HIGH)", minimum=0.0, maximum=1.0, value=0.2, step=0.1)

    submit = gr.Button("Generate Script for me")

    # Outputs
    title_output = gr.Textbox(label="Title 🔥")
    script_output = gr.Textbox(label="Your Video Script 📝", lines=15)
    search_output = gr.Textbox(label="DuckDuckGo Search Results 🔍", lines=5)

    submit.click(
        fn=generate,
        inputs=[prompt, video_length, creativity, api_key],
        outputs=[title_output, script_output, search_output]
    )

if __name__ == "__main__":
    demo.launch()