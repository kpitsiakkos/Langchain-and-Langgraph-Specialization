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
