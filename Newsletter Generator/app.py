import gradio as gr
from helpers import (
    search_serp,
    pick_best_articles_urls,
    extract_content_from_urls,
    summarizer,
    generate_newsletter,
)

css = """
/* ── Page ───────────────────────────────────────────────── */
body, .gradio-container {
    background: #f4f1eb !important;
    font-family: 'Georgia', serif !important;
}

/* ── Header ─────────────────────────────────────────────── */
.prose h1 {
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
    letter-spacing: -0.5px;
    border-bottom: 3px double #1a1a1a;
    padding-bottom: 0.5rem;
    margin-bottom: 0.25rem !important;
}
.prose p {
    color: #444 !important;
    font-size: 0.95rem !important;
    font-style: italic;
}

/* ── Input ───────────────────────────────────────────────── */
.gr-box, label.svelte-1b6s6s, .gr-input-label {
    font-family: 'Georgia', serif !important;
}
input[type="text"], textarea {
    background: #fffef9 !important;
    border: 1px solid #c8c0a8 !important;
    border-radius: 4px !important;
    font-family: 'Georgia', serif !important;
    font-size: 0.95rem !important;
    color: #1a1a1a !important;
    padding: 10px 14px !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #8a7a5a !important;
    box-shadow: 0 0 0 2px rgba(138, 122, 90, 0.15) !important;
    outline: none !important;
}

/* ── Button ──────────────────────────────────────────────── */
.gr-button-primary, button.primary {
    background: #c0392b !important;
    color: #fff !important;
    font-family: 'Georgia', serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 12px 28px !important;
    transition: background 0.2s ease;
}
.gr-button-primary:hover, button.primary:hover {
    background: #96281b !important;
}

/* ── Accordions ──────────────────────────────────────────── */
.gr-accordion {
    background: #fffef9 !important;
    border: 1px solid #d4cbb8 !important;
    border-radius: 4px !important;
    margin-bottom: 10px !important;
}
.gr-accordion > .label-wrap {
    padding: 12px 16px !important;
    font-family: 'Georgia', serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    color: #1a1a1a !important;
    border-bottom: 1px solid #e0d8c8;
}
.gr-accordion > .label-wrap:hover {
    background: #f5f0e8 !important;
}

/* ── Textboxes inside accordions ─────────────────────────── */
.gr-accordion textarea {
    background: #fffef9 !important;
    border: 1px solid #e0d8c8 !important;
    border-radius: 4px !important;
    font-family: 'Georgia', serif !important;
    font-size: 0.88rem !important;
    color: #333 !important;
    line-height: 1.7 !important;
}

/* ── Newsletter accordion — highlighted ──────────────────── */
.newsletter-block .gr-accordion {
    border: 2px solid #1a1a1a !important;
    background: #fffef9 !important;
}
.newsletter-block textarea {
    font-size: 0.95rem !important;
    line-height: 1.8 !important;
    color: #1a1a1a !important;
}

/* ── Labels ──────────────────────────────────────────────── */
label span, .gr-input-label span {
    font-family: 'Georgia', serif !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #777 !important;
}
"""


def run_pipeline(query):
    if not query or not query.strip():
        return ("", "", "", "", "")

    search_results    = search_serp(query=query)
    urls              = pick_best_articles_urls(response_json=search_results, query=query)
    data              = extract_content_from_urls(urls)
    summaries         = summarizer(data, query)
    newsletter_thread = generate_newsletter(summaries, query)

    urls_text = "\n".join(urls) if isinstance(urls, list) else str(urls)
    data_text = " ".join(d.page_content for d in data.similarity_search(query, k=4))

    return (
        str(search_results),
        urls_text,
        data_text,
        str(summaries),
        str(newsletter_thread),
    )


with gr.Blocks(title="Newsletter Generator 🦜", css=css) as demo:

    gr.Markdown("# 🦜 Generate a Newsletter")
    gr.Markdown("Enter a topic and the app will search, summarise, and write a newsletter thread for you.")

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

    with gr.Group(elem_classes="newsletter-block"):
        with gr.Accordion("📰 Newsletter", open=True):
            newsletter_out = gr.Textbox(label="Final newsletter thread", lines=12, interactive=False)

    generate_btn.click(
        fn=run_pipeline,
        inputs=[query_input],
        outputs=[search_out, urls_out, data_out, summaries_out, newsletter_out],
    )


if __name__ == "__main__":
    demo.launch()