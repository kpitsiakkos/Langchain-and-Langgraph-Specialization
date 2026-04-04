from io import StringIO
from dotenv import load_dotenv
import gradio as gr
import anthropic
import base64
import time

# Load environment variables from .env
load_dotenv()

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from .env


def review_python_file(file):
    if file is None:
        return "Please upload a .py file to get started.", ""

    try:
        with open(file.name, "r", encoding="utf-8") as f:
            fetched_data = f.read()
    except Exception as e:
        return f"Error reading file: {e}", ""

    # Call Claude for code review
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=(
                "You are a code review assistant. "
                "Provide detailed suggestions to improve the given Python code. "
                "Structure your review with sections: Summary, Issues Found, "
                "Suggestions, and Improved Code if applicable. Use markdown formatting."
            ),
            messages=[
                {"role": "user", "content": fetched_data}
            ],
        )
        review = message.content[0].text
    except Exception as e:
        return fetched_data, f"Error calling Claude API: {e}"

    return fetched_data, review


def download_review(review_text):
    if not review_text:
        return None
    timestr = time.strftime("%Y%m%d-%H%M%S")
    filename = f"/tmp/code_review_analysis_file_{timestr}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(review_text)
    return filename


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 Let's do code review for your Python code")
    gr.Markdown("### Please upload your `.py` file here:")

    file_input = gr.File(label="Upload Python file", file_types=[".py"])

    with gr.Row():
        with gr.Column():
            file_content = gr.Code(
                label="📄 File Content",
                language="python",
                lines=20,
                interactive=False,
            )
        with gr.Column():
            review_output = gr.Markdown(label="🤖 Claude's Code Review")

    review_btn = gr.Button("🚀 Run Code Review", variant="primary")
    download_btn = gr.DownloadButton("⬇️ Download Review", visible=False)

    # Auto-load file content on upload
    file_input.change(
        fn=lambda f: open(f.name).read() if f else "",
        inputs=file_input,
        outputs=file_content,
    )

    # Run review on button click
    def run_and_show_download(file):
        content, review = review_python_file(file)
        return content, review, gr.update(visible=bool(review))

    review_btn.click(
        fn=run_and_show_download,
        inputs=file_input,
        outputs=[file_content, review_output, download_btn],
    )

    # Wire download button
    review_output.change(
        fn=download_review,
        inputs=review_output,
        outputs=download_btn,
    )

demo.launch()