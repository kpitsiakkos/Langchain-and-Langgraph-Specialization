import gradio as gr
from dotenv import load_dotenv

load_dotenv()


def analyze_csv(file, query):
    # Get Response
    return "sharath"


with gr.Blocks(title="CSV Analysis") as demo:
    gr.Markdown("# Let's do some analysis on your CSV")
    gr.Markdown("## Please upload your CSV file here:")

    # Capture the CSV file
    data = gr.File(label="Upload CSV file", file_types=[".csv"])
    query = gr.Textbox(label="Enter your query", lines=3)
    button = gr.Button("Generate Response")
    output = gr.Textbox(label="Response", lines=10)

    button.click(
        fn=analyze_csv,
        inputs=[data, query],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()