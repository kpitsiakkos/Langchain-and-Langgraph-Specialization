import gradio as gr
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to load the Llama 2 model using CTransformers
# CTransformers provides Python bindings for transformer models 
# This allows us to run quantized models efficiently on CPU
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

# Function to build the prompt template using LangChain's PromptTemplate
# This structures the input before sending it to the model
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

# Main function to generate the email response
# Takes user inputs, formats the prompt, and invokes the LLM
def generate_email(email_topic, sender_name, recipient_name, writing_style):
    if not email_topic or not sender_name or not recipient_name:
        return "Please fill in all fields before generating."
    
    llm = load_llm()
    prompt = build_prompt()

    # Format the prompt with user inputs and invoke the model
    response = llm.invoke(prompt.format(
        email_topic=email_topic,
        sender=sender_name,
        recipient=recipient_name,
        style=writing_style
    ))

    return response

# Building the Gradio UI using Blocks for a structured layout
with gr.Blocks(title="Generate Emails") as app:
    gr.Markdown("# Generate Emails 📧")

    # Text area for the user to enter the email topic
    form_input = gr.Textbox(label="Enter the email topic", lines=10)

    # Row layout for sender, recipient, and style inputs
    with gr.Row():
        email_sender = gr.Textbox(label="Sender Name")
        email_recipient = gr.Textbox(label="Recipient Name")
        email_style = gr.Dropdown(
            choices=["Formal", "Appreciating", "Not Satisfied", "Neutral"],
            value="Formal",
            label="Writing Style"
        )

    # Generate button triggers the email generation function
    submit = gr.Button("Generate")

    # Output textbox to display the generated email
    output = gr.Textbox(label="Generated Email", lines=15)

    # Linking the button click to the generate_email function
    submit.click(
        fn=generate_email,
        inputs=[form_input, email_sender, email_recipient, email_style],
        outputs=output
    )

app.launch()