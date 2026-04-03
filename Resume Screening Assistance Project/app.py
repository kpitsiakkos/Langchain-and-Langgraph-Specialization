import gradio as gr
from dotenv import load_dotenv
from utils import *
import uuid
import os

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:         #f7f4ef;
    --white:      #ffffff;
    --surface:    #faf8f5;
    --border:     #e8e2d9;
    --accent:     #2d6a4f;
    --accent-lt:  #e8f4ee;
    --accent-dk:  #1b4332;
    --ink:        #1a1a1a;
    --muted:      #888880;
    --warning:    #e76f51;
    --radius:     8px;
    --shadow:     0 2px 16px rgba(0,0,0,0.07);
    --font-body:  'Inter', sans-serif;
    --font-head:  'Playfair Display', serif;
}

html, body, .gradio-container {
    background: var(--bg) !important;
    font-family: var(--font-body) !important;
    color: var(--ink) !important;
}

.gradio-container {
    max-width: 880px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

footer { display: none !important; }

/* ── Hero ── */
.hero-wrap {
    background: var(--accent-dk);
    padding: 52px 40px 44px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.03);
}
.hero-wrap h1 {
    font-family: var(--font-head) !important;
    font-size: 2.6rem;
    font-weight: 600;
    color: #ffffff !important;
    margin: 0 0 10px;
    line-height: 1.15;
    position: relative; z-index: 1;
}
.hero-wrap h1 span {
    font-style: italic;
    font-weight: 400;
    color: #74c69d;
}
.hero-wrap p {
    color: rgba(255,255,255,0.55);
    font-size: 0.82rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0;
    position: relative; z-index: 1;
}

/* ── Instructions ── */
.instr-wrap {
    background: var(--accent-lt);
    border: 1px solid #b7dfc8;
    border-radius: var(--radius);
    padding: 20px 24px;
    margin: 0 32px 28px;
}
.instr-wrap h3 {
    font-family: var(--font-head) !important;
    color: var(--accent-dk);
    font-size: 1rem;
    font-weight: 600;
    margin: 0 0 10px;
}
.instr-wrap ol {
    margin: 0; padding-left: 18px;
    color: #3a5a47; font-size: 0.875rem; line-height: 1.95;
}
.instr-wrap ol li strong { color: var(--accent-dk); }

/* ── Form wrapper ── */
.form-wrap {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px;
    margin: 0 32px 24px;
    box-shadow: var(--shadow);
}

/* ── Gradio block resets ── */
.gradio-container .block,
.gradio-container .panel,
.gradio-container .form,
.gradio-container .gap,
.gradio-container > .gradio-container {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ── Labels ── */
.gradio-container label span,
.gradio-container .label-wrap span {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin-bottom: 6px !important;
}

/* ── Inputs ── */
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"] {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--ink) !important;
    font-family: var(--font-body) !important;
    font-size: 0.93rem !important;
    padding: 11px 14px !important;
    transition: border-color 0.18s, box-shadow 0.18s !important;
}
.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(45,106,79,0.1) !important;
    outline: none !important;
    background: var(--white) !important;
}
.gradio-container textarea::placeholder,
.gradio-container input::placeholder { color: #bbb !important; }

/* ── File upload ── */
.gradio-container .upload-container,
.gradio-container [data-testid="file-upload"],
.gradio-container .file-preview-holder,
.gradio-container .wrap.svelte-iyf88w {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--muted) !important;
    transition: border-color 0.18s !important;
}
.gradio-container .upload-container:hover {
    border-color: var(--accent) !important;
    background: var(--accent-lt) !important;
}

/* ── Button ── */
.gradio-container button.primary,
.gradio-container .gr-button-primary {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-body) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 14px 28px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.18s, transform 0.15s, box-shadow 0.18s !important;
    margin-top: 6px !important;
}
.gradio-container button.primary:hover {
    background: var(--accent-dk) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(27,67,50,0.25) !important;
}

/* ── Results ── */
.results-md {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 28px 32px !important;
    margin: 0 32px 16px !important;
    box-shadow: var(--shadow) !important;
    min-height: 100px !important;
    color: var(--ink) !important;
}
.results-md h3 {
    font-family: var(--font-head) !important;
    color: var(--accent-dk) !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    margin: 8px 0 4px !important;
}
.results-md code {
    background: var(--accent-lt) !important;
    color: var(--accent-dk) !important;
    padding: 2px 8px !important;
    border-radius: 4px !important;
    font-size: 0.84rem !important;
}
.results-md hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 18px 0 !important;
}
.results-md p { color: #444 !important; line-height: 1.75 !important; }

/* ── Status ── */
.status-wrap { margin: 0 32px 32px !important; }
.status-wrap textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--muted) !important;
    font-size: 0.85rem !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    color: #bbb;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 20px 0 40px;
}
"""

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
        embeddings = create_embeddings_load_data()

        push_to_pinecone(PINECONE_API_KEY, pinecone_index_name.strip(), embeddings, final_docs_list)

        relevant_docs = similar_docs(
            job_description, document_count, PINECONE_API_KEY,
            pinecone_index_name.strip(), embeddings, unique_id
        )

        output = f"### {len(final_docs_list)} resumes analysed - top {document_count} matches\n\n---\n\n"
        medals = ["🥇", "🥈", "🥉"]
        for item in range(len(relevant_docs)):
            medal = medals[item] if item < 3 else f"**#{item+1}**"
            output += f"### {medal} {relevant_docs[item][0].metadata['name']}\n\n"
            score = round(float(relevant_docs[item][1]), 4)
            output += f"**Match Score:** `{score}`\n\n"
            summary = get_summary(relevant_docs[item][0])
            output += f"**Summary:** {summary}\n\n"
            output += "---\n\n"

        return output, "✅ Analysis complete - hope I saved you some time! ❤️"

    except Exception as e:
        return f"❌ An error occurred: {str(e)}", ""


with gr.Blocks(css=css, title="HR Resume Screening", theme=gr.themes.Base(
    primary_hue=gr.themes.colors.green,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
)) as demo:

    gr.HTML("""
    <div class="hero-wrap">
        <h1>Resume <span>Screening</span> Assistant</h1>
        <p>AI-powered candidate ranking &nbsp;·&nbsp; LangChain &amp; Pinecone</p>
    </div>
    <div class="instr-wrap">
        <h3>How to use</h3>
        <ol>
            <li><strong>Paste the Job Description</strong> - include required skills, experience level, and responsibilities.</li>
            <li><strong>Set how many results</strong> - enter how many top candidates you want returned (e.g. <strong>3</strong>).</li>
            <li><strong>Enter your Pinecone index name</strong> - use an existing index from your Pinecone dashboard.</li>
            <li><strong>Upload resumes</strong> - drag &amp; drop or click to select multiple PDF files at once.</li>
            <li><strong>Click Analyse</strong> - candidates are ranked by match score with an AI-generated summary for each.</li>
        </ol>
    </div>
    """)

    with gr.Column(elem_classes=["form-wrap"]):
        job_description = gr.Textbox(
            label="Job Description",
            placeholder="Paste the full job description here - skills, responsibilities, requirements...",
            lines=7
        )
        with gr.Row():
            document_count = gr.Textbox(
                label="Number of Resumes to Return",
                placeholder="e.g. 3",
                scale=1
            )
            pinecone_index_name = gr.Textbox(
                label="Pinecone Index Name",
                placeholder="e.g. tickets, resumescreening",
                scale=2
            )
        pdf_files = gr.File(
            label="Upload Resumes - PDF only",
            file_types=[".pdf"],
            file_count="multiple"
        )
        submit_btn = gr.Button("Analyse Resumes", variant="primary")

    results_output = gr.Markdown(label="Results", elem_classes=["results-md"])
    status_output = gr.Textbox(label="Status", interactive=False, elem_classes=["status-wrap"])

    gr.HTML('<div class="app-footer">Resume Screening Assistant &nbsp;·&nbsp; LangChain · Pinecone · Gradio</div>')

    submit_btn.click(
        fn=analyze_resumes,
        inputs=[job_description, document_count, pinecone_index_name, pdf_files],
        outputs=[results_output, status_output]
    )

if __name__ == "__main__":
    print("Starting Gradio app...")
    demo.launch(server_name="0.0.0.0", share=False)