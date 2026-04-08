import gradio as gr
import pandas as pd
from helpers import *

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

* {
    font-family: 'DM Sans', sans-serif;
}

body, .gradio-container {
    background-color: #f5f3ef !important;
}

/* Header */
.gradio-container h1 {
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: #1a1a1a !important;
    letter-spacing: -0.5px;
    padding: 1.5rem 0 0.5rem;
}

/* Upload box */
.upload-box {
    border: 1.5px dashed #c9c3ba !important;
    border-radius: 12px !important;
    background: #ffffff !important;
    transition: border-color 0.2s ease;
}
.upload-box:hover {
    border-color: #f59e0b !important;
}

/* Extract button */
button.primary {
    background: #1a1a1a !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 12px 28px !important;
    border: none !important;
    transition: background 0.2s ease, transform 0.1s ease;
}
button.primary:hover {
    background: #f59e0b !important;
    color: #1a1a1a !important;
    transform: translateY(-1px);
}

/* Labels */
label span, .block > label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #888 !important;
}

/* Table */
.dataframe table {
    border-collapse: collapse !important;
    background: #ffffff !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}
.dataframe thead tr {
    background: #f0ede8 !important;
    color: #555 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
.dataframe td, .dataframe th {
    padding: 10px 14px !important;
    border-bottom: 1px solid #f0ede8 !important;
    color: #2a2a2a !important;
}
.dataframe tr:last-child td {
    border-bottom: none !important;
}
.dataframe tr:hover td {
    background: #fffbf4 !important;
}

/* Summary textbox */
textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    color: #1a1a1a !important;
    background: #ffffff !important;
    border: 1.5px solid #e8e4de !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
}

/* Download file area */
.file-preview {
    background: #ffffff !important;
    border: 1.5px solid #e8e4de !important;
    border-radius: 10px !important;
}

/* Panels / blocks */
.block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
"""

def extract_bills(pdf_files):
    if not pdf_files:
        return None, "Please upload at least one PDF file.", None

    data_frame = create_docs(pdf_files)
    data_frame["AMOUNT"] = data_frame["AMOUNT"].str.replace(",", "").astype(float)
    average = data_frame["AMOUNT"].mean()

    csv_path = "CSV_Bills.csv"
    data_frame.to_csv(csv_path, index=False)

    summary = f"Average bill amount: ${average:,.2f}"

    return data_frame, summary, csv_path


with gr.Blocks(title="Bill Extractor", css=custom_css) as app:
    gr.Markdown("# Bill Extractor AI Assistant 🤖")

    pdf_input = gr.File(
        label="Upload your bills in PDF format only",
        file_types=[".pdf"],
        file_count="multiple",
        elem_classes=["upload-box"]
    )

    extract_button = gr.Button("Extract bill data...", variant="primary")

    output_table = gr.Dataframe(label="Extracted Bill Data")
    output_summary = gr.Textbox(label="Summary")
    output_csv = gr.File(label="Download data as CSV")

    extract_button.click(
        fn=extract_bills,
        inputs=[pdf_input],
        outputs=[output_table, output_summary, output_csv]
    )

if __name__ == "__main__":
    app.launch()