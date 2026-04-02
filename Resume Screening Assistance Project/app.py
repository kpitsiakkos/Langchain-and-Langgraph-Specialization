import gradio as gr
from dotenv import load_dotenv
from utils import *
import uuid
import os

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def analyze_resumes(job_description, document_count, pinecone_index_name, pdf_files):
    if not job_description.strip():
        return "❌ Please provide a job description.", ""
    if not pdf_files:
        return "❌ Please upload at least one resume PDF.", ""
    if not document_count.strip().isdigit():
        return "❌ Please enter a valid number of resumes to return.", ""
    if not pinecone_index_name.strip():
        return "❌ Please provide a Pinecone index name.", ""

    try:
        unique_id = uuid.uuid4().hex

        final_docs_list = create_docs(pdf_files, unique_id)
        resume_count_msg = f"✅ **Resumes uploaded:** {len(final_docs_list)}\n\n"

        embeddings = create_embeddings_load_data()

        push_to_pinecone(PINECONE_API_KEY, pinecone_index_name.strip(), embeddings, final_docs_list)

        relevant_docs = similar_docs(
            job_description,
            document_count,
            PINECONE_API_KEY,
            pinecone_index_name.strip(),
            embeddings,
            unique_id
        )

        output = resume_count_msg + "---\n\n"

        for item in range(len(relevant_docs)):
            output += f"### 👉 {item + 1}\n"
            output += f"**File:** {relevant_docs[item][0].metadata['name']}\n\n"
            output += f"**Match Score:** {relevant_docs[item][1]}\n\n"
            summary = get_summary(relevant_docs[item][0])
            output += f"**Summary:** {summary}\n\n"
            output += "---\n\n"

        return output, "Hope I was able to save your time ❤️"

    except Exception as e:
        return f"❌ An error occurred: {str(e)}", ""


with gr.Blocks(title="Resume Screening Assistance") as demo:
    gr.Markdown("# HR - Resume Screening Assistance 💁")
    gr.Markdown("I can help you in the resume screening process.")

    with gr.Row():
        with gr.Column():
            job_description = gr.Textbox(
                label="Job Description",
                placeholder="Please paste the 'JOB DESCRIPTION' here...",
                lines=8
            )
            document_count = gr.Textbox(
                label="Number of Resumes to Return",
                placeholder="e.g. 3"
            )
            pinecone_index_name = gr.Textbox(
                label="Pinecone Index Name",
                placeholder="e.g. tickets, websitechatbot, mcqcreator",
                value="resumescreening"
            )
            pdf_files = gr.File(
                label="Upload Resumes (PDF only)",
                file_types=[".pdf"],
                file_count="multiple"
            )
            submit_btn = gr.Button("Help me with the analysis", variant="primary")

    with gr.Row():
        results_output = gr.Markdown(label="Results")

    with gr.Row():
        status_output = gr.Textbox(label="Status", interactive=False)

    submit_btn.click(
        fn=analyze_resumes,
        inputs=[job_description, document_count, pinecone_index_name, pdf_files],
        outputs=[results_output, status_output]
    )

if __name__ == "__main__":
    print("Starting Gradio app...")
    demo.launch(server_name="0.0.0.0", share=False)