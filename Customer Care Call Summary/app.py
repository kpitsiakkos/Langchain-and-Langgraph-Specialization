import gradio as gr
from utils import email_summary
import os

css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #F0F4FF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #1E1B4B !important;
    min-height: 100vh;
}

.gradio-container > .main > .wrap {
    max-width: 760px !important;
    margin: 0 auto !important;
    padding: 0 20px 80px !important;
}

#hero {
    text-align: center;
    padding: 52px 0 36px;
}
#hero .badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #EEF2FF;
    border: 1.5px solid #C7D2FE;
    color: #4338CA;
    font-size: 11px; font-weight: 700;
    padding: 5px 14px; border-radius: 99px;
    margin-bottom: 18px; letter-spacing: .1em;
    text-transform: uppercase;
}
#hero .dot {
    width: 7px; height: 7px;
    border-radius: 50%; background: #10B981;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50% { opacity:.4; transform:scale(1.4); }
}
#hero h1 {
    font-size: 40px !important;
    font-weight: 800 !important;
    letter-spacing: -.03em !important;
    line-height: 1.15 !important;
    color: #1E1B4B !important;
    margin-bottom: 10px !important;
}
#hero h1 span { color: #4F46E5; }
#hero p { font-size: 15px; color: #6B7280; line-height: 1.6; }

.stat-row {
    display: grid; grid-template-columns: repeat(3,1fr); gap: 12px;
    margin-bottom: 20px;
}
.stat {
    border-radius: 16px; padding: 18px 20px;
    font-weight: 700;
}
.stat-ai   { background: #EEF2FF; border: 2px solid #C7D2FE; }
.stat-stt  { background: #ECFDF5; border: 2px solid #A7F3D0; }
.stat-zap  { background: #FFF7ED; border: 2px solid #FED7AA; }
.stat-val  { font-size: 22px; font-weight: 800; }
.stat-ai   .stat-val { color: #4F46E5; }
.stat-stt  .stat-val { color: #059669; }
.stat-zap  .stat-val { color: #EA580C; }
.stat-lbl  { font-size: 10px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; margin-top: 3px; color: #9CA3AF; }

.section-lbl {
    font-size: 10px !important; font-weight: 700 !important;
    letter-spacing: .1em !important; text-transform: uppercase !important;
    color: #9CA3AF !important; margin-bottom: 10px !important; display: block;
}

#upload-zone {
    background: #fff !important;
    border: 2px solid #E0E7FF !important;
    border-radius: 18px !important;
    padding: 20px !important; margin-bottom: 14px !important;
    transition: border-color .2s !important;
}
#upload-zone:hover { border-color: #818CF8 !important; }
#upload-zone .wrap {
    border: 2px dashed #C7D2FE !important;
    border-radius: 12px !important;
    background: #F5F3FF !important;
    min-height: 130px !important;
    transition: all .2s !important;
}
#upload-zone .wrap:hover {
    border-color: #4F46E5 !important;
    background: #EEF2FF !important;
}
#upload-zone .icon-wrap svg { color: #818CF8 !important; }
#upload-zone .label { font-size: 13px !important; color: #6B7280 !important; font-weight: 500 !important; }

#email-input {
    background: #fff !important;
    border: 2px solid #E0E7FF !important;
    border-radius: 18px !important;
    padding: 20px !important; margin-bottom: 14px !important;
}
#email-input input {
    background: #F9FAFB !important;
    border: 1.5px solid #E5E7EB !important;
    border-radius: 10px !important;
    color: #1E1B4B !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    padding: 11px 16px !important;
    height: 44px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
#email-input input:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 4px rgba(79,70,229,0.12) !important;
    outline: none !important;
    background: #fff !important;
}
#email-input input::placeholder { color: #D1D5DB !important; }

#send-btn {
    background: #4F46E5 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    height: 48px !important;
    cursor: pointer !important;
    transition: background .15s, transform .1s, box-shadow .15s !important;
    box-shadow: 0 4px 14px rgba(79,70,229,0.35) !important;
}
#send-btn:hover {
    background: #4338CA !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(79,70,229,0.4) !important;
}
#send-btn:active { transform: translateY(0) !important; }

#clear-btn {
    background: #fff !important;
    color: #9CA3AF !important;
    border: 1.5px solid #E5E7EB !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    height: 48px !important;
    cursor: pointer !important;
    transition: all .15s !important;
}
#clear-btn:hover {
    background: #FFF1F2 !important;
    color: #F43F5E !important;
    border-color: #FECDD3 !important;
}

#log-box {
    background: #fff !important;
    border: 2px solid #E0E7FF !important;
    border-radius: 18px !important;
    padding: 20px !important; margin-top: 14px !important;
}
#log-box textarea {
    background: #F0FDF4 !important;
    border: 1.5px solid #BBF7D0 !important;
    border-radius: 10px !important;
    color: #065F46 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.8 !important;
    resize: none !important;
    min-height: 110px !important;
    padding: 14px !important;
}

.tab-nav { border-bottom: 2px solid #E0E7FF !important; margin-bottom: 20px !important; }
.tab-nav button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 600 !important;
    color: #9CA3AF !important;
    border-bottom: 3px solid transparent !important;
    background: transparent !important;
    padding: 10px 4px !important; margin-right: 24px !important;
    transition: color .15s !important;
}
.tab-nav button.selected {
    color: #4F46E5 !important;
    border-bottom-color: #4F46E5 !important;
}

.gr-markdown h2 { font-size: 16px !important; font-weight: 700 !important; color: #1E1B4B !important; margin: 1.5rem 0 .6rem !important; }
.gr-markdown p, .gr-markdown li { font-size: 14px !important; color: #6B7280 !important; line-height: 1.7 !important; }
.gr-markdown code { background: #EEF2FF !important; color: #4338CA !important; padding: 2px 6px !important; border-radius: 5px !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }

#footer {
    text-align: center; padding: 24px 0 0;
    font-size: 11px; color: #D1D5DB;
    letter-spacing: .08em; text-transform: uppercase;
    border-top: 2px solid #E0E7FF; margin-top: 36px;
}

label { color: #9CA3AF !important; font-size: 11px !important; }
"""

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

with gr.Blocks(css=css, title="Customer Care · Call Summarizer") as demo:

    gr.HTML("""
    <div id="hero">
        <div class="badge"><span class="dot"></span>Live</div>
        <h1>Customer care<br><span>call summarizer</span></h1>
        <p>Upload call recordings &middot; Auto-transcribe with Whisper &middot; Email summaries via Zapier</p>
    </div>
    <div class="stat-row">
        <div class="stat stat-ai"><div class="stat-val">GPT</div><div class="stat-lbl">GPT-3.5 Turbo</div></div>
        <div class="stat stat-stt"><div class="stat-val">STT</div><div class="stat-lbl">Whisper Base</div></div>
        <div class="stat stat-zap"><div class="stat-val">ZAP</div><div class="stat-lbl">Zapier Webhook</div></div>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("Upload & Send"):

            gr.HTML('<p class="section-lbl">Upload recordings</p>')
            audio_input = gr.File(label="", file_types=[".mp3"], file_count="multiple", elem_id="upload-zone")

            gr.HTML('<p class="section-lbl" style="margin-top:4px;">Recipient email</p>')
            email_input = gr.Textbox(placeholder="customer-care@company.com", label="", elem_id="email-input")

            with gr.Row():
                send_btn  = gr.Button("↗  Transcribe & send summary", elem_id="send-btn",  scale=4)
                clear_btn = gr.Button("Clear all",                     elem_id="clear-btn", scale=1)

            gr.HTML('<p class="section-lbl" style="margin-top:8px;">Activity log</p>')
            log_box = gr.Textbox(value="Ready.", label="", interactive=False, lines=5, elem_id="log-box")

            send_btn.click(fn=process_and_send, inputs=[audio_input, email_input], outputs=log_box)
            clear_btn.click(fn=clear_all, inputs=[], outputs=[audio_input, email_input, log_box])

        with gr.Tab("How it works"):
            gr.Markdown("""
## Pipeline

1. **Upload** one or more `.mp3` call recordings.
2. **Enter** the destination email address.
3. Click **Transcribe & send** - the app will:
   - Transcribe each file locally using **OpenAI Whisper** (base model).
   - Summarise the transcript using **GPT-3.5 Turbo** via the OpenAI API.
   - Trigger a **Zapier webhook** which fires a Gmail send action.

---

## Configuration

Set the following in your `.env` file:

```
OPENAI_API_KEY=sk-...
ZAPIER_WEBHOOK_URL=https://hooks.zapier.com/hooks/catch/...
```
""")

    gr.HTML('<div id="footer">Customer Care Intelligence &middot; Whisper &middot; OpenAI &middot; Zapier</div>')

if __name__ == "__main__":
    demo.launch()