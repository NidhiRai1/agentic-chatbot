# Agentic Chatbot

An intelligent, agent-based chatbot designed to generate interactive responses by leveraging both predefined FAQs and dynamic PDF document analysis. Built with Python, this project integrates multiple components to deliver context-aware and informative interactions.

## Features

- **Agent-Based Architecture**: Modular design with specialized agents.
- **FAQ Integration**: Responds to predefined frequently asked questions.
- **PDF Analysis**: Extracts insights from uploaded PDF documents.
- **Interactive Frontend**: Simple interface for user interaction.
- **Backend Logic**: Coordinates agents and handles requests efficiently.

##Output 

![Chatbot Screenshot](Screenshot (126).png)

## Project Structure

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
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install dependencies
Backend python.py

# Install dependencies
streamlit run frontend.py

