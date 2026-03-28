import gradio as gr
from langchain_openai import OpenAI
from langchain_classic.chains import ConversationChain
from langchain_classic.memory import ConversationSummaryMemory

# ============================================================
# GLOBAL STATE
# Holds the active conversation chain across interactions
# ============================================================
conversation = None

# ============================================================
# CORE FUNCTION: Get LLM Response
# Initializes the conversation chain on first use,
# then predicts a response based on user input
# ============================================================
def getresponse(user_input, api_key):
    global conversation

    # Guard: API key must be provided
    if not api_key:
        return "⚠️ Please enter your API key in the field above.", []

    # Guard: User must type something
    if not user_input.strip():
        return "⚠️ Please enter a message before sending.", []

    # Initialize conversation chain only once per session
    if conversation is None:
        llm = OpenAI(
            temperature=0,
            openai_api_key=api_key,
            model_name='gpt-3.5-turbo-instruct'  # text-davinci-003 is deprecated
        )
        # ConversationSummaryMemory: LLM summarizes history to stay within token limits
        conversation = ConversationChain(
            llm=llm,
            verbose=True,
            memory=ConversationSummaryMemory(llm=llm)
        )

    # Get response from LLM
    response = conversation.predict(input=user_input)

    # Debug: Print current memory buffer to terminal
    print("📝 Memory Buffer:\n", conversation.memory.buffer)

    return response
# ============================================================
# SUMMARISE FUNCTION
# Returns the current summarized conversation memory
# ============================================================
def summarise():
    if conversation is None:
        return "💬 No conversation to summarise yet. Start chatting first!"
    return "Nice chatting with you my friend ❤️\n\n" + conversation.memory.buffer

# ============================================================
# RESET FUNCTION
# Clears the conversation and resets UI
# ============================================================
def reset():
    global conversation
    conversation = None
    return [], ""

# ============================================================
# CHAT HANDLER
# Appends user message and AI response to chat history
# ============================================================
def chat(user_input, api_key, history):
    response = getresponse(user_input, api_key)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return history, ""

# ============================================================
# GRADIO UI
# ============================================================
with gr.Blocks() as app:

    # ── Header ──────────────────────────────────────────────
    gr.Markdown("<h1 style='text-align: center;'>🤖 ChatGPT Clone</h1>")
    gr.Markdown("<p class='subtitle'>Powered by LangChain + OpenAI | Uses Summary Memory to stay within token limits</p>")

    # ── API Key Input ────────────────────────────────────────
    gr.Markdown("### 🔑 Step 1: Enter your OpenAI API Key")
    gr.Markdown("<small style='color: #888;'>Your key is never stored — it's used only for this session.</small>")
    api_key = gr.Textbox(
        label="OpenAI API Key",
        type="password",
        placeholder="sk-...",
        elem_classes="api-box"
    )

    # ── Chat Area ────────────────────────────────────────────
    gr.Markdown("### 💬 Step 2: Start Chatting")
    chatbot = gr.Chatbot(
        label="Conversation",
        height=420,
        type="messages",
        elem_classes="chatbot"
    )

    # ── User Input ───────────────────────────────────────────
    user_input = gr.Textbox(
        label="Your message",
        placeholder="Type your question here and click Send...",
        lines=3
    )

    # ── Action Buttons ───────────────────────────────────────
    gr.Markdown("### ⚡ Actions")
    with gr.Row():
        submit = gr.Button("📨 Send", elem_id="send-btn")
        summarise_btn = gr.Button("📋 Summarise Conversation", elem_id="summarise-btn")
        reset_btn = gr.Button("🔄 Reset Chat", elem_id="reset-btn")

    # ── Summary Output ───────────────────────────────────────
    gr.Markdown("### 📝 Conversation Summary")
    gr.Markdown("<small style='color: #888;'>Click 'Summarise' to see what the AI remembers about your conversation.</small>")
    summary_output = gr.Textbox(
        label="Summary",
        lines=6,
        interactive=False,
        placeholder="Your conversation summary will appear here...",
        elem_id="summary-box"
    )

    # ── Footer ───────────────────────────────────────────────
    gr.Markdown("<hr><p style='text-align:center; color:#aaa; font-size:0.8rem;'>Built with 🦜 LangChain + 🤗 Gradio</p>")

    # ── Event Handlers ───────────────────────────────────────
    submit.click(
        fn=chat,
        inputs=[user_input, api_key, chatbot],
        outputs=[chatbot, user_input]
    )
    summarise_btn.click(
        fn=summarise,
        outputs=summary_output
    )
    reset_btn.click(
        fn=reset,
        outputs=[chatbot, summary_output]
    )

# ── Launch ───────────────────────────────────────────────────
app.launch(

theme=gr.themes.Soft(),  # Clean soft theme for better readability
    css="""
        /* Page background */
        .gradio-container {
            background-color: #f0f4f8;
        }

        /* Main title */
        h1 {
            color: #1a1a2e;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 5px;
        }

        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #555;
            font-size: 0.95rem;
            margin-bottom: 20px;
        }

        /* API key box highlight */
        .api-box textarea {
            border: 2px solid #4a90d9 !important;
            border-radius: 8px !important;
        }

        /* Send button — green */
        #send-btn {
            background-color: #28a745 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
        }

        /* Summarise button — blue */
        #summarise-btn {
            background-color: #007bff !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
        }

        /* Reset button — red */
        #reset-btn {
            background-color: #dc3545 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
        }

        /* Summary output box */
        #summary-box textarea {
            background-color: #fff8e1 !important;
            border: 2px solid #ffc107 !important;
            border-radius: 8px;
            color: #333;
        }

        /* Chat bubble area */
        .chatbot {
            border-radius: 12px;
            background-color: #ffffff;
        }
    """
)