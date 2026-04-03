import gradio as gr
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers

def load_llm():
    llm = CTransformers(
        model='/Users/kpitsiakkos/Documents/Langchain-and-Langgraph-Specialization/Email Generator /llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={
            'max_new_tokens': 256,
            'temperature': 0.01
        }
    )
    return llm

def build_prompt():
    template = """
    Write an email with {style} style and includes topic: {email_topic}.\n\nSender: {sender}\nRecipient: {recipient}
    \n\nEmail Text:
    """
    prompt = PromptTemplate(
        input_variables=["style", "email_topic", "sender", "recipient"],
        template=template
    )
    return prompt

def generate_email(email_topic, sender_name, recipient_name, writing_style):
    if not email_topic or not sender_name or not recipient_name:
        return "Please fill in all fields before generating."
    llm = load_llm()
    prompt = build_prompt()
    response = llm.invoke(prompt.format(
        email_topic=email_topic,
        sender=sender_name,
        recipient=recipient_name,
        style=writing_style
    ))
    return response

css = """
    * {
        box-sizing: border-box;
    }

    body, .gradio-container {
        background: #0d0d0d !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }

    .gradio-container {
        max-width: 820px !important;
        margin: 50px auto !important;
        padding: 0 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
    }

    /* Title */
    #title {
        text-align: center !important;
        padding: 40px 0 10px 0 !important;
    }

    #title h1 {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        color: #ffffff !important;
        letter-spacing: -1px !important;
        margin: 0 !important;
    }

    #title h1 span {
        background: linear-gradient(90deg, #6366f1, #ec4899) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }

    #subtitle {
        text-align: center !important;
        margin-bottom: 36px !important;
    }

    #subtitle p {
        color: #6b7280 !important;
        font-size: 1rem !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        font-weight: 400 !important;
    }

    /* Blocks / Cards */
    .block, .form {
        background: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 16px !important;
        padding: 20px !important;
        margin-bottom: 16px !important;
    }

    /* Labels */
    label span {
        color: #9ca3af !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
    }

    /* Inputs */
    textarea, input[type="text"] {
        background: #111111 !important;
        color: #f3f4f6 !important;
        border: 1px solid #2e2e2e !important;
        border-radius: 10px !important;
        font-size: 0.97rem !important;
        padding: 12px 14px !important;
        caret-color: #6366f1 !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    }

    textarea:focus, input[type="text"]:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
        outline: none !important;
    }

    textarea::placeholder, input::placeholder {
        color: #4b5563 !important;
    }

    /* Dropdown */
    .wrap, select {
        background: #111111 !important;
        color: #f3f4f6 !important;
        border: 1px solid #2e2e2e !important;
        border-radius: 10px !important;
        font-size: 0.97rem !important;
    }

    /* Generate Button */
    button.primary, button {
        background: linear-gradient(135deg, #6366f1 0%, #ec4899 100%) !important;
        color: #ffffff !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 24px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.3s ease !important;
        cursor: pointer !important;
    }

    button.primary:hover, button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(236, 72, 153, 0.4) !important;
    }

    /* Output box */
    #output textarea {
        background: #111111 !important;
        color: #d1fae5 !important;
        border: 1px solid #064e3b !important;
        border-radius: 12px !important;
        font-size: 0.97rem !important;
        line-height: 1.7 !important;
    }

    /* Hide footer */
    footer {
        display: none !important;
    }
"""

with gr.Blocks(title="Generate Emails", css=css) as app:
    gr.Markdown("# ✉️ Generate <span>Emails</span>", elem_id="title")
    gr.Markdown("Powered by Llama 2 running locally on your machine", elem_id="subtitle")

    form_input = gr.Textbox(
        label="Email Topic",
        placeholder="Describe what your email should be about...",
        lines=8
    )

    with gr.Row():
        email_sender = gr.Textbox(label="Sender Name", placeholder="e.g. John Smith")
        email_recipient = gr.Textbox(label="Recipient Name", placeholder="e.g. Sarah Johnson")
        email_style = gr.Dropdown(
            choices=["Formal", "Appreciating", "Not Satisfied", "Neutral"],
            value="Formal",
            label="Writing Style"
        )

    submit = gr.Button("⚡ Generate Email")

    output = gr.Textbox(
        label="Generated Email",
        lines=15,
        elem_id="output"
    )

    submit.click(
        fn=generate_email,
        inputs=[form_input, email_sender, email_recipient, email_style],
        outputs=output
    )

app.launch()