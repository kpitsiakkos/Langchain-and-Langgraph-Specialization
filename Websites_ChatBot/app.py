import gradio as gr
import os

# ── Load external CSS ─────────────────────────────────────────────────────────

with open("/Users/kpitsiakkos/Documents/Langchain-and-Langgraph-Specialization/Websites_ChatBot/style.css", "r") as f:
    custom_css = f.read()


# ── Stub functions — replace with real utils.py calls ─────────────────────────

def load_data(hf_key: str, pinecone_key: str):
    if not hf_key.strip() or not pinecone_key.strip():
        return "⚠️ Please provide both API keys."

    os.environ["PINECONE_API_KEY"] = pinecone_key

    # get_website_data() → split_data() → create_embeddings() → push_to_pinecone()
    return "✅ Data pushed to Pinecone successfully!"


def search(hf_key: str, pinecone_key: str, query: str, doc_count: int):
    if not hf_key.strip() or not pinecone_key.strip():
        return "⚠️ Please provide both API keys."
    if not query.strip():
        return "⚠️ Please enter a question."

    os.environ["PINECONE_API_KEY"] = pinecone_key

    # create_embeddings() → pull_from_pinecone() → get_similar_docs()
    return (
        "### 👉 Result 1\n"
        "**Info:** Sample result — replace with document.page_content\n\n"
        "**Link:** \n\n---\n\n"
        "### 👉 Result 2\n"
        "**Info:** Sample result — replace with document.page_content\n\n"
        "**Link:** \n\n---\n\n"
        "### 👉 Result 3\n"
        "**Info:** Sample result — replace with document.page_content\n\n"
        "**Link:** \n\n---\n\n"
    )


# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(css=custom_css, title="🤖 AI Assistance For Website") as demo:

    gr.Markdown("# 🤖 AI Assistance For Website")

    with gr.Row():

        # ── Left column — API keys & load ─────────────────────────────────────
        with gr.Column(scale=1):

            gr.Markdown("### 😎🗝️")

            hf_key = gr.Textbox(
                label="What's your HuggingFace API key?",
                type="password",
                placeholder="hf_...",
            )
            pinecone_key = gr.Textbox(
                label="What's your Pinecone API key?",
                type="password",
                placeholder="xxxxxxxx-xxxx-...",
            )

            load_btn = gr.Button("Load data to Pinecone", variant="primary")

            load_status = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Status will appear here...",
            )

            load_btn.click(
                fn=load_data,
                inputs=[hf_key, pinecone_key],
                outputs=load_status,
            )

        # ── Right column — search ─────────────────────────────────────────────
        with gr.Column(scale=3):

            query = gr.Textbox(
                label="How can I help you my friend ❓",
                placeholder="Ask me anything...",
                lines=2,
            )

            doc_count = gr.Slider(
                minimum=0,
                maximum=5,
                value=2,
                step=1,
                label="No. Of links to return 🔗 - (0 LOW || 5 HIGH)",
            )

            search_btn = gr.Button("Search", variant="primary")

            results = gr.Markdown(value="")

            search_btn.click(
                fn=search,
                inputs=[hf_key, pinecone_key, query, doc_count],
                outputs=results,
            )


if __name__ == "__main__":
    demo.launch()