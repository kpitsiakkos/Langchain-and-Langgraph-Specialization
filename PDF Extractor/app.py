import gradio as gr
import pandas as pd


def extract_bills(pdf_files):
    if not pdf_files:
        return None, "Please upload at least one PDF file.", None

    data_frame = create_docs(pdf_files)
    data_frame["AMOUNT"] = data_frame["AMOUNT"].astype(float)
    average = data_frame["AMOUNT"].mean()

    # Save to CSV
    csv_path = "CSV_Bills.csv"
    data_frame.to_csv(csv_path, index=False)

    summary = f"Average bill amount: ${average:.2f}"

    return data_frame, summary, csv_path


with gr.Blocks(title="Bill Extractor") as app:
    gr.Markdown("# Bill Extractor AI Assistant 🤖")

    pdf_input = gr.File(
        label="Upload your bills in PDF format only",
        file_types=[".pdf"],
        file_count="multiple"
    )

    extract_button = gr.Button("Extract bill data...")

    with gr.Column(visible=True):
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