# LangChain & LangGraph Specialization

This repository tracks my progress through the **LangChain and LangGraph Specialization** course on Coursera, offered by Packt.

**Course Link**: [LangChain and LangGraph Specialization](https://www.coursera.org/specializations/packt-langchain-and-langgraph)

---

## About the Course

This specialization covers building real-world AI applications using LangChain and LangGraph frameworks. Topics include:

- Building conversational AI applications with LLMs
- Working with OpenAI embeddings and vector similarity
- Creating RAG (Retrieval-Augmented Generation) systems
- Building AI agents with LangGraph
- Deploying AI applications to production

---

## Repository Structure

Each notebook corresponds to a module in the course:

| File | Description |
|------|-------------|
| `libraries.ipynb` | Setting up the environment and installing dependencies |
| `Chat_Model_intro.ipynb` | Introduction to ChatOpenAI and message types |
| `Huggingface_token_setup.ipynb` | Setting up HuggingFace authentication |
| `app.py` | Gradio web app demo using LangChain |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/kpitsiakkos/Langchain-and-Langgraph-Specialization.git
cd Langchain-and-Langgraph-Specialization
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## API Keys & Tokens

This project requires two API keys. **Never hardcode them in your code or commit them to GitHub.**

### OpenAI API Key
- Sign up at [platform.openai.com](https://platform.openai.com)
- Go to **API Keys** and create a new key
- Create a file called `openai_token.txt` in the project root and paste your key inside
- This file is listed in `.gitignore` and will **never** be pushed to GitHub

### HuggingFace Token
- Sign up at [huggingface.co](https://huggingface.co)
- Go to **Settings -> Access Tokens** and create a new token
- Create a file called `huggingface_token.txt` in the project root and paste your token inside
- This file is also listed in `.gitignore`

### For Hugging Face Spaces Deployment
If deploying to Hugging Face Spaces, add your keys as Secrets:
1. Go to your Space -> **Settings -> Variables and Secrets**
2. Add `OPENAI_API_KEY` as a secret
3. Add `HUGGINGFACE_TOKEN` as a secret
4. Access them in code with `os.getenv("OPENAI_API_KEY")`

---

## Goals

- [x] Learn LangChain basics
- [x] Build a chat model with OpenAI
- [x] Work with embeddings and similarity search
- [ ] Build RAG systems
- [ ] Build AI agents with LangGraph
- [ ] Deploy real-world AI applications

---

## Disclaimer

All notebooks and code in this repository are based on materials from the Coursera course by Packt. They are for personal learning purposes only.