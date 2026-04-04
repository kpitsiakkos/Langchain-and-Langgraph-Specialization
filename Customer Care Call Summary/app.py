import gradio as gr
from utils import email_summary
import os

# ── CSS ───────────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

body, .gradio-container {
    background: #f5f5f3 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container > .main > .wrap {
    max-width: 680px !important;
    margin: 0 auto !important;
    padding: 0 16px 64px !important;
}

#hero {
    text-align: center;
    padding: 48px 0 32px;
}
#hero h1 {
    font-size: 28px !important;
    font-weight: 600 !important;
    color: #0f0f0f !important;
    line-height: 1.25 !important;
    margin-bottom: 8px !important;
}
#hero p { font-size: 14px; color: #6b7280; }
#hero .badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #dbeafe; color: #1e40af;
    font-size: 11px; font-weight: 500;
    padding: 4px 12px; border-radius: 99px;
    margin-bottom: 14px; letter-spacing: .05em;
}

.gr-group, .gr-box {
    background: #fff !important;
    border: 0.5px solid rgba(0,0,0,0.08) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin-bottom: 12px !important;
    box-shadow: none !important;
}
.section-lbl {
    font-size: 10px !important; font-weight: 600 !important;
    letter-spacing: .08em !important; text-transform: uppercase !important;
    color: #9ca3af !important; margin-bottom: 10px !important; display: block;
}

#upload-zone {
    background: #fff !important;
    border: 0.5px solid rgba(0,0,0,0.08) !important;
    border-radius: 12px !important;
    padding: 20px !important; margin-bottom: 12px !important;
}
#upload-zone .wrap {
    border: 1.5px dashed rgba(0,0,0,0.15) !important;
    border-radius: 10px !important; background: transparent !important;
    min-height: 130px !important; transition: border-color .2s, background .2s !important;
}
#upload-zone .wrap:hover {
    border-color: #3b82f6 !important; background: #eff6ff !important;
}
#upload-zone .icon-wrap svg { color: #9ca3af !important; }
#upload-zone .label { font-size: 13px !important; color: #374151 !important; font-weight: 500 !important; }

#email-input {
    background: #fff !important;
    border: 0.5px solid rgba(0,0,0,0.08) !important;
    border-radius: 12px !important;
    padding: 20px !important; margin-bottom: 12px !important;
}
#email-input input {
    border: 0.5px solid rgba(0,0,0,0.12) !important;
    border-radius: 8px !important; background: #fafafa !important;
    color: #0f0f0f !important; font-family: 'Inter', sans-serif !important;
    font-size: 14px !important; padding: 10px 14px !important; height: 40px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
#email-input input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1) !important;
    outline: none !important; background: #fff !important;
}
#email-input input::placeholder { color: #9ca3af !important; }

#send-btn {
    background: #0f0f0f !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important; font-weight: 500 !important;
    padding: 10px 20px !important; height: 42px !important;
    cursor: pointer !important; transition: opacity .15s !important;
}
#send-btn:hover { opacity: .82 !important; }

#clear-btn {
    background: transparent !important; color: #6b7280 !important;
    border: 0.5px solid rgba(0,0,0,0.15) !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important; font-weight: 400 !important;
    padding: 10px 18px !important; height: 42px !important;
    cursor: pointer !important; transition: background .15s !important;
}
#clear-btn:hover { background: #f3f4f6 !important; }

#log-box {
    background: #fff !important;
    border: 0.5px solid rgba(0,0,0,0.08) !important;
    border-radius: 12px !important; padding: 20px !important; margin-top: 12px !important;
}
#log-box textarea {
    background: #f9fafb !important;
    border: 0.5px solid rgba(0,0,0,0.08) !important;
    border-radius: 8px !important; color: #374151 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important; line-height: 1.7 !important;
    resize: none !important; min-height: 100px !important; padding: 12px !important;
}

.tab-nav { border-bottom: 0.5px solid rgba(0,0,0,0.1) !important; margin-bottom: 16px !important; }
.tab-nav button {
    font-family: 'Inter', sans-serif !important; font-size: 13px !important;
    font-weight: 500 !important; color: #9ca3af !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important; padding: 8px 4px !important;
    margin-right: 20px !important; transition: color .15s !important;
}
.tab-nav button.selected { color: #0f0f0f !important; border-bottom-color: #0f0f0f !important; }

#footer {
    text-align: center; padding: 24px 0 0;
    font-size: 11px; color: #d1d5db;
    letter-spacing: .06em; text-transform: uppercase;
    border-top: 0.5px solid rgba(0,0,0,0.06); margin-top: 32px;
}
"""

# ── Logic ─────────────────────────────────────────────────────────────────────
def process_and_send(audio_files, recipient_email):
    if not audio_files:
        return "⚠  No files uploaded. Please upload at least one MP3."
    if not recipient_email or "@" not in recipient_email:
        return "⚠  Please enter a valid recipient email address."

    log_lines = []
    for file_obj in audio_files:
        file_path = file_obj.name
        file_name = os.path.basename(file_path)
        log_lines.append(f"⏳  Transcribing: {file_name} …")
        try:
            email_summary(file_path, recipient_email)
            log_lines.append(f"✓   Email sent for: {file_name}")
        except Exception as e:
            log_lines.append(f"✕   Failed - {file_name}: {str(e)}")

    return "\n".join(log_lines)


def clear_all():
    return None, "", "Ready."


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="Customer Care · Call Summarizer") as demo:

    gr.HTML("""
    <div id="hero">
        <div class="badge">&#x25CF; Live</div>
        <h1>Customer care<br>call summarizer</h1>
        <p>Upload call recordings &middot; Auto-transcribe &middot; Email summaries instantly</p>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("Upload & Send"):

            gr.HTML('<p class="section-lbl">Upload recordings</p>')
            audio_input = gr.File(
                label="",
                file_types=[".mp3"],
                file_count="multiple",
                elem_id="upload-zone",
            )

            gr.HTML('<p class="section-lbl" style="margin-top:4px;">Recipient email</p>')
            email_input = gr.Textbox(
                placeholder="customer-care@company.com",
                label="",
                elem_id="email-input",
            )

            with gr.Row():
                send_btn  = gr.Button("↗  Transcribe & send summary", elem_id="send-btn",  scale=4)
                clear_btn = gr.Button("Clear all",                     elem_id="clear-btn", scale=1)

            gr.HTML('<p class="section-lbl" style="margin-top:8px;">Activity log</p>')
            log_box = gr.Textbox(
                value="Ready.",
                label="",
                interactive=False,
                lines=5,
                elem_id="log-box",
            )

            send_btn.click(fn=process_and_send, inputs=[audio_input, email_input], outputs=log_box)
            clear_btn.click(fn=clear_all, inputs=[], outputs=[audio_input, email_input, log_box])

        with gr.Tab("How it works"):
            gr.Markdown("""
## Pipeline

1. **Upload** one or more `.mp3` call recordings.
2. **Enter** the destination email address.
3. Click **Transcribe & send** - the app will:
   - Transcribe each file locally using **OpenAI Whisper** (base model).
   - Summarise the transcript using a **LangChain + OpenAI** agent.
   - Deliver the summary via a **Zapier Gmail** action.

---

## Configuration

Set the following environment variables before launching:

```
OPENAI_API_KEY=sk-...
ZAPIER_NLA_API_KEY=sk-ak-...
```

All audio processing happens **locally** - files are never sent to a third-party transcription service.
""")

    gr.HTML('<div id="footer">Customer Care Intelligence &middot; Whisper &middot; LangChain &middot; Zapier</div>')


if __name__ == "__main__":
    demo.launch()