from io import StringIO
from dotenv import load_dotenv
import gradio as gr

# Load environment variables from .env
load_dotenv()

def read_python_file(file):
    if file is None:
        return "Please upload your .py file here."

    try:
        with open(file.name, "r", encoding="utf-8") as f:
            data = f.read()

        stringio = StringIO(data)
        fetched_data = stringio.read()

        return fetched_data

    except Exception as e:
        return f"Error reading file: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# Let's do code review for your python code")
    gr.Markdown("### Please upload your .py file here:")

    file_input = gr.File(label="Upload python file", file_types=[".py"])
    output_box = gr.Textbox(label="File Content", lines=25)

    file_input.change(fn=read_python_file, inputs=file_input, outputs=output_box)

demo.launch()