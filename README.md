# 🦜 LangChain & LangGraph Specialization

This repository tracks my progress through the **LangChain and LangGraph Specialization** on Coursera, offered by Packt.

**Course Link**: [LangChain and LangGraph Specialization](https://www.coursera.org/specializations/packt-langchain-and-langgraph)

---

## About the Course

This specialization covers building real-world AI applications using the LangChain and LangGraph frameworks. Topics include:

- Building conversational AI applications with LLMs
- Working with OpenAI embeddings and vector similarity
- Creating RAG (Retrieval-Augmented Generation) systems
- Building AI agents with LangGraph
- Deploying AI applications to production

---

## Repository Structure

### Root-Level Notebooks & Files

| File | Description |
|---|---|
| `chains.ipynb` | Foundational chain concepts and patterns |
| `chain_story.ipynb` | Sequential chain storytelling demo |
| `Chat_Model_intro.ipynb` | Introduction to ChatOpenAI and message types |
| `router_chain.py` | Multi-subject router chain implementation |
| `libraries.ipynb` | Environment setup and dependency overview |
| `Huggingface_token_setup.ipynb` | HuggingFace API token configuration |
| `Text_Embedding_example.ipynb` | Text embeddings with OpenAI |
| `word_embeddings.csv` | Sample data for embedding experiments |
| `CSV_Bills.csv` / `data.xlsx` | Sample datasets used across projects |

### Projects

| Project | Description |
|---|---|
| **Agents** | Autonomous agent implementations using LangChain's agent framework with tool use and decision-making loops |
| **Automatic Ticket Classification Tool** | Classifies incoming support tickets by category and priority using an LLM chain |
| **ChatGPT Clone** | Conversational chatbot with persistent memory, replicating a ChatGPT-style experience with a Gradio UI |
| **Code Review Analysis** | Submits code snippets to an LLM for automated review, surfacing bugs, style issues, and improvement suggestions |
| **CSV Data Analysis** | Natural language interface for querying and summarising CSV data using LangChain's CSV agent |
| **Customer Care Call Summary** | Transcribes audio with Whisper, summarises with GPT, and delivers the result via a Zapier webhook email |
| **Data Connection Module** | Demonstrates LangChain's document loaders and data connectors for integrating external sources |
| **Email Generator** | Generates professional, context-aware emails from brief prompts using a prompt template chain |
| **Image to Text App** | Extracts and interprets text or content from images using a vision-capable LLM pipeline |
| **Invoice Extractor Bot** | Parses invoice PDFs or images and extracts structured fields (vendor, amount, date) |
| **Marketing Campaign App** | Generates multi-channel marketing copy (social, email, ad) from a product description |
| **MultiDocument Chatbot** | RAG-based chatbot that ingests multiple documents and answers queries via vector store retrieval |
| **Multiple Choice Question Creator App** | Automatically generates MCQs from any input text for study materials or quiz generation |
| **Newsletter Generator** | Produces formatted newsletter content from a topic using chained summarisation and copywriting prompts |
| **PDF Extractor** | Extracts and structures key information from PDF documents using document loaders and LLM-powered parsing |
| **Resume Screening Assistance Project** | Screens resumes against a job description, scoring and summarising candidate fit |
| **Text to SQL Query** | Translates natural language questions into SQL queries and executes them against a database |
| **Translator App** | Multi-language translation app with a Gradio interface, powered by an LLM prompt chain |
| **Websites ChatBot** | Crawls and indexes a website's content, then answers questions about it via a RAG pipeline |
| **YouTube Script Writing Tool** | Generates full YouTube video scripts from a title and topic using sequential chains |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Orchestration | LangChain (LCEL), LangGraph |
| LLM Providers | OpenAI (GPT-3.5 / GPT-4), Anthropic Claude |
| Speech | OpenAI Whisper |
| Embeddings | OpenAI, HuggingFace |
| Vector Store | FAISS, Chroma |
| UI | Gradio |
| Automation | Zapier Webhooks |
| Language | Python 3.10 |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/kpitsiakkos/Langchain-and-Langgraph-Specialization.git
cd Langchain-and-Langgraph-Specialization
```

### 2. Create and activate a virtual environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run a project

Each project folder contains its own `app.py` or notebook. For Gradio apps:
```bash
cd "Email Generator"
python app.py
```

---

## API Keys & Tokens

This project requires multiple API keys depending on the project. **Never hardcode them in your code or commit them to GitHub.**

Create a `.env` file in the project root with the following variables:

```env
# LLM Providers
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Vector Store
PINECONE_API_KEY=your_pinecone_key

# Open Source Models
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
REPLICATE_API_TOKEN=your_replicate_token

# Automation & Webhooks
ZAPIER_NLA_API_KEY=your_zapier_nla_key
ZAPIER_WEBHOOK_URL=your_zapier_webhook_url

# Search
SERPER_API_KEY=your_serper_key
TAVILY_API_KEY=your_tavily_key
```

> `.env` is listed in `.gitignore` and will **never** be pushed to GitHub.

### Where to get each key

| Key | Provider | Link |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | [platform.openai.com](https://platform.openai.com) → API Keys |
| `ANTHROPIC_API_KEY` | Anthropic | [console.anthropic.com](https://console.anthropic.com) → API Keys |
| `PINECONE_API_KEY` | Pinecone | [app.pinecone.io](https://app.pinecone.io) → API Keys |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace | [huggingface.co](https://huggingface.co) → Settings → Access Tokens |
| `REPLICATE_API_TOKEN` | Replicate | [replicate.com](https://replicate.com) → Account → API Tokens |
| `ZAPIER_NLA_API_KEY` | Zapier | [nla.zapier.com](https://nla.zapier.com) → Settings |
| `ZAPIER_WEBHOOK_URL` | Zapier | Create a Zap with a Webhook trigger |
| `SERPER_API_KEY` | Serper | [serper.dev](https://serper.dev) → Dashboard |
| `TAVILY_API_KEY` | Tavily | [tavily.com](https://tavily.com) → Dashboard |

### For Hugging Face Spaces Deployment
1. Go to your Space → **Settings → Variables and Secrets**
2. Add each key above as a secret
3. Access them in code with `os.getenv("OPENAI_API_KEY")`

---

## Progress

- [x] Learn LangChain basics
- [x] Build a chat model with OpenAI
- [x] Work with embeddings and similarity search
- [x] Build RAG systems
- [x] Build AI agents
- LangChain & LangGraph Specialization

This repository tracks my progress through the **LangChain and LangGraph Specialization** on Coursera, offered by Packt.

**Course Link**: [LangChain and LangGraph Specialization](https://www.coursera.org/specializations/packt-langchain-and-langgraph)

---

## About the Course

This specialization covers building real-world AI applications using the LangChain and LangGraph frameworks. Topics include:

- Building conversational AI applications with LLMs
- Working with OpenAI embeddings and vector similarity
- Creating RAG (Retrieval-Augmented Generation) systems
- Building AI agents with LangGraph
- Deploying AI applications to production

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/kpitsiakkos/Langchain-and-Langgraph-Specialization.git
cd Langchain-and-Langgraph-Specialization
```

### 2. Create and activate a virtual environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run a project

Each project folder contains its own `app.py` or notebook. For Gradio apps:
```bash
cd "Email Generator"
python app.py
```

---

## API Keys & Tokens

This project requires multiple API keys depending on the project. **Never hardcode them in your code or commit them to GitHub.**

Create a `.env` file in the project root with the following variables:

```env
# LLM Providers
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Vector Store
PINECONE_API_KEY=your_pinecone_key

# Open Source Models
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
REPLICATE_API_TOKEN=your_replicate_token

# Automation & Webhooks
ZAPIER_NLA_API_KEY=your_zapier_nla_key
ZAPIER_WEBHOOK_URL=your_zapier_webhook_url

# Search
SERPER_API_KEY=your_serper_key
TAVILY_API_KEY=your_tavily_key
```

> `.env` is listed in `.gitignore` and will **never** be pushed to GitHub.

### Where to get each key

| Key | Provider | Link |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | [platform.openai.com](https://platform.openai.com) → API Keys |
| `ANTHROPIC_API_KEY` | Anthropic | [console.anthropic.com](https://console.anthropic.com) → API Keys |
| `PINECONE_API_KEY` | Pinecone | [app.pinecone.io](https://app.pinecone.io) → API Keys |
| `HUGGINGFACEHUB_API_TOKEN` | HuggingFace | [huggingface.co](https://huggingface.co) → Settings → Access Tokens |
| `REPLICATE_API_TOKEN` | Replicate | [replicate.com](https://replicate.com) → Account → API Tokens |
| `ZAPIER_NLA_API_KEY` | Zapier | [nla.zapier.com](https://nla.zapier.com) → Settings |
| `ZAPIER_WEBHOOK_URL` | Zapier | Create a Zap with a Webhook trigger |
| `SERPER_API_KEY` | Serper | [serper.dev](https://serper.dev) → Dashboard |
| `TAVILY_API_KEY` | Tavily | [tavily.com](https://tavily.com) → Dashboard |

### For Hugging Face Spaces Deployment
1. Go to your Space → **Settings → Variables and Secrets**
2. Add each key above as a secret
3. Access them in code with `os.getenv("OPENAI_API_KEY")`

---

## Progress

- [x] Learn LangChain basics
- [x] Build a chat model with OpenAI
- [x] Work with embeddings and similarity search
- [x] Build RAG systems
- [x] Build AI agents
- [x] Build AI agents with LangGraph
- [x] Deploy real-world AI applications

---

## Notes

- All chains use modern **LCEL pipe syntax** (`prompt | llm | parser`) — no deprecated `LLMChain` or `SequentialChain`.
- Imports follow current conventions: `langchain_openai`, `langchain_core`, `langchain_community`.
- Gradio is used consistently for UI across projects.

---

## Disclaimer

All notebooks and code in this repository are based on materials from the Coursera course by Packt. They are for personal learning purposes only.] Build AI agents with LangGraph
- [ ] Deploy real-world AI applications

---

## Notes

- All chains use modern **LCEL pipe syntax** (`prompt | llm | parser`) — no deprecated `LLMChain` or `SequentialChain`.
- Imports follow current conventions: `langchain_openai`, `langchain_core`, `langchain_community`.
- Gradio is used consistently for UI across projects.

---

## Disclaimer

All notebooks and code in this repository are based on materials from the Coursera course by Packt. They are for personal learning purposes only.