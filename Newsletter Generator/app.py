import gradio as gr

# ── Placeholder — swap these out with your real helpers later ──────────────
def search_serp(query): return f"[search results for '{query}']"
def pick_best_articles_urls(response_json, query): return []
def extract_content_from_urls(urls): return None          # will be your FAISS db
def summarizer(data, query): return f"[summaries for '{query}']"
def generate_newsletter(summaries, query): return f"[newsletter thread for '{query}']"
# ──────────────────────────────────────────────────────────────────────────

def run_pipeline(query):
    if not query or not query.strip():
        return ("", "", "", "", "")

    search_results       = search_serp(query=query)
    urls                 = pick_best_articles_urls(response_json=search_results, query=query)
    data                 = extract_content_from_urls(urls)
    summaries            = summarizer(data, query)
    newsletter_thread    = generate_newsletter(summaries, query)

    urls_text  = "\n".join(urls) if isinstance(urls, list) else str(urls)
    data_text  = (
        " ".join(d.page_content for d in data.similarity_search(query, k=4))
        if data is not None else "(data not available yet)"
    )

    return (
        str(search_results),
        urls_text,
        data_text,
        str(summaries),
        str(newsletter_thread),
    )


with gr.Blocks(title="Newsletter Generator 🦜") as demo:

    gr.Markdown("# 🦜 Generate a Newsletter")
    gr.Markdown("Enter a topic below and the app will search, summarise, and write a newsletter thread for you.")

    query_input = gr.Textbox(
        label="Topic",
        placeholder="e.g. Flutter development news",
        lines=1,
    )

    generate_btn = gr.Button("Generate Newsletter", variant="primary")

    with gr.Accordion("🔍 Search Results", open=False):
        search_out = gr.Textbox(label="Raw search results", lines=6, interactive=False)

    with gr.Accordion("🔗 Best URLs", open=False):
        urls_out = gr.Textbox(label="Selected article URLs", lines=4, interactive=False)

    with gr.Accordion("📄 Extracted Data", open=False):
        data_out = gr.Textbox(label="Top-k similarity chunks", lines=6, interactive=False)

    with gr.Accordion("📝 Summaries", open=False):
        summaries_out = gr.Textbox(label="Per-article summaries", lines=6, interactive=False)

    with gr.Accordion("📰 Newsletter", open=True):
        newsletter_out = gr.Textbox(label="Final newsletter thread", lines=12, interactive=False)

    generate_btn.click(
        fn=run_pipeline,
        inputs=[query_input],
        outputs=[search_out, urls_out, data_out, summaries_out, newsletter_out],
    )

if __name__ == "__main__":
    demo.launch()