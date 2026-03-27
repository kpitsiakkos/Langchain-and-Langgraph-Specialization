import gradio as gr
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# load_dotenv() loads variables from a .env file into environment variables
from dotenv import load_dotenv
load_dotenv()
# Initialize the OpenAIEmbeddings object
embeddings = OpenAIEmbeddings()

# Import CSV file data
from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path='myData.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})

# Assigning the data inside the csv to our variable
data = loader.load()
print(data)

db = FAISS.from_documents(data, embeddings)

# Function to find similar things based on user input
def find_similar(user_input):
    if not user_input:
        return "Please enter something to search."
    
    docs = db.similarity_search(user_input)
    print(docs)
    
    result = "Top Matches:\n\n"
    result += f"Match 1: {docs[0].page_content}\n\n"
    result += f"Match 2: {docs[1].page_content}"
    return result

# Build Gradio UI
with gr.Blocks() as app:
    gr.Markdown("# Hey, Ask me something & I will give out similar things")
    
    user_input = gr.Textbox(label="You:", placeholder="Type something...")
    submit = gr.Button("Find Similar Things")
    output = gr.Textbox(label="Top Matches:")
    
    submit.click(fn=find_similar, inputs=user_input, outputs=output)

app.launch()