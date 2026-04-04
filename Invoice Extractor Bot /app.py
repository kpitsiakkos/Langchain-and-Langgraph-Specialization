import gradio as gr
from dotenv import load_dotenv
from utils import create_docs


def extract_invoices(pdf_files):
    if not pdf_files:
        return None, "Please upload at least one PDF invoice."
    
    df = create_docs(pdf_files)

    if df.empty:
        return None, "No data could be extracted from the uploaded invoices."

    csv_path = "invoice_data.csv"
    df.to_csv(csv_path, index=False)

    return df, csv_path


with gr.Blocks(title="Invoice Extraction Bot") as app:
    gr.Markdown("# Invoice Extraction Bot")
    gr.Markdown("Upload one or more PDF invoices to extract structured data and download as CSV.")

    with gr.Row():
        pdf_input = gr.File(
            label="Upload PDF invoices",
            file_types=[".pdf"],
            file_count="multiple"
        )

    extract_btn = gr.Button("Extract Data", variant="primary")

    with gr.Row():
        status_msg = gr.Textbox(label="Status", interactive=False, visible=False)

    data_table = gr.Dataframe(label="Extracted Invoice Data", interactive=False)
    csv_download = gr.File(label="Download CSV", visible=False)

    def run_extraction(pdf_files):
        if not pdf_files:
            return (
                gr.update(value="Please upload at least one PDF.", visible=True),
                None,
                gr.update(visible=False)
            )
        try:
            df, csv_path = extract_invoices(pdf_files)
            return (
                gr.update(value="Extraction complete!", visible=True),
                df,
                gr.update(value=csv_path, visible=True)
            )
        except Exception as e:
            return (
                gr.update(value=f"Error: {str(e)}", visible=True),
                None,
                gr.update(visible=False)
            )

    extract_btn.click(
        fn=run_extraction,
        inputs=[pdf_input],
        outputs=[status_msg, data_table, csv_download]
    )


if __name__ == "__main__":
    load_dotenv()
    app.launch()
