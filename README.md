# 🤖 LangGraph AI Chatbot Agent

A modular Conversational AI system powered by LangGraph and LangChain that supports advanced ReAct agents, OCR-based image inputs, web and academic search tools, and PDF report generation. The backend is built using FastAPI, and the frontend is powered by Streamlit.

## Features

- ReAct Agent with LangGraph for reasoning and tool-use
- Supports both **Groq** and **OpenAI** LLM providers
- Web Search with **Tavily**
- Academic Paper Search with **arXiv**
- Optional **PDF generation** of AI responses
- OCR: Extract text from uploaded images and use it in queries
- Session memory with chat history
- FAQ Fallback for predefined queries
- Streamlit UI for interactive use

## Architecture

agentic-chatbot/
├── ai_agent.py # Core reasoning and response logic

├── backend.py # Backend server logic

├── frontend.py # User interface script

├── faq_data.py # FAQ dataset

├── pdf_tool.py # PDF extraction and processing

├── requirements.txt # Dependencies

├── README.md # Project documentation

├── .gitignore # Git ignore rules

├── venv/ # Python virtual environment

├── pdfs/ # Directory for PDF documents

└── pycache/ # Compiled cache files


## Installation

### Prerequisites

- Python 3.8+
- Git

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/NidhiRai1/agentic-chatbot.git
cd agentic-chatbot

# Set up a virtual environment
python -m venv venv
.\venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt

# Install dependencies
Backend python.py

# Install dependencies
streamlit run frontend.py

