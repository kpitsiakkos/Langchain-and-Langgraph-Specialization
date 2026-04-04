import gradio as gr
from dotenv import load_dotenv
from utils import create_docs
import os

extracted_df = None

css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
with open(css_path, "r") as f:
    custom_css = f.read()


def extract_invoices(pdf_files):
    global extracted_df

    if not pdf_files:
        return (
            gr.update(value="Please upload at least one PDF.", visible=True),
            None,
            gr.update(visible=False),
            gr.update(visible=False)
        )
    try:
        extracted_df = create_docs(pdf_files)

        if extracted_df.empty:
            return (
                gr.update(value="No data could be extracted.", visible=True),
                None,
                gr.update(visible=False),
                gr.update(visible=False)
            )

        return (
            gr.update(value="✅  Extraction complete!", visible=True),
            extracted_df,
            gr.update(visible=True),
            gr.update(visible=False)
        )
    except Exception as e:
        return (
            gr.update(value=f"❌  Error: {str(e)}", visible=True),
            None,
            gr.update(visible=False),
            gr.update(visible=False)
        )


def download_csv():
    global extracted_df

    if extracted_df is None or extracted_df.empty:
        return gr.update(visible=False)

    csv_path = "invoice_data.csv"
    extracted_df.to_csv(csv_path, index=False)
    return gr.update(value=csv_path, visible=True)


with gr.Blocks(title="Invoice Extraction Bot", css=custom_css) as app:

    with gr.Column(elem_id="header-block"):
        gr.Markdown("# 🧾 Invoice Extraction Bot")
        gr.Markdown("Upload one or more PDF invoices to extract structured data into a CSV.")

    with gr.Column(elem_id="upload-zone"):
        pdf_input = gr.File(
            label="Upload PDF invoices",
            file_types=[".pdf"],
            file_count="multiple"
        )

    extract_btn = gr.Button("Extract Data", variant="primary", elem_id="extract-btn")

    status_msg = gr.Textbox(
        label="",
        interactive=False,
        visible=False,
        elem_id="status-box"
    )

    data_table = gr.Dataframe(
        label="Extracted Invoice Data",
        interactive=False,
        wrap=True,
        elem_id="data-table"
    )

    download_btn = gr.Button("⬇  Download as CSV", variant="secondary", visible=False, elem_id="download-btn")
    csv_download = gr.File(label="Your CSV is ready", visible=False, elem_id="csv-output")

    extract_btn.click(
        fn=extract_invoices,
        inputs=[pdf_input],
        outputs=[status_msg, data_table, download_btn, csv_download]
    )

    download_btn.click(
        fn=download_csv,
        inputs=[],
        outputs=[csv_download]
    )


if __name__ == "__main__":
    load_dotenv()
    app.launch()