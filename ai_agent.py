# ai_agent.py

from dotenv import load_dotenv
load_dotenv()

import os

# Step 1: Load API keys from environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Step 2: Import LangChain LLMs and Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun, ArxivAPIWrapper
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

# Step 3: Import PDF generator
from pdf_tool import generate_pdf_report


def get_response_from_ai_agent(
    llm_id: str,
    query: list,
    allow_search: bool,
    allow_arxiv: bool,
    allow_pdf: bool,
    system_prompt: str,
    provider: str
) -> dict:
    """
    Run a LangGraph ReAct agent with optional web search, arXiv support, and PDF output.

    Args:
        llm_id (str): Model ID (e.g., "llama3-70b-8192")
        query (list): List of formatted chat messages (e.g., ["User: hi", "AI: hello"])
        allow_search (bool): Whether to include the web search tool (Tavily)
        allow_arxiv (bool): Whether to include the arXiv academic search tool
        allow_pdf (bool): Whether to generate a downloadable PDF from the AI response
        system_prompt (str): Instruction to guide the agent's behavior
        provider (str): "Groq" or "OpenAI"

    Returns:
        dict: {"response": final_response_text, "pdf_path": path_to_pdf (if enabled)}
    """

    # Step 1: Initialize LLM based on provider
    provider = provider.lower()
    if provider == "groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "openai":
        llm = ChatOpenAI(model=llm_id)
    else:
        raise ValueError("Unsupported provider. Use 'Groq' or 'OpenAI'.")

    # Step 2: Dynamically assemble toolset
    tools = []

    if allow_search:
        tools.append(TavilySearchResults(max_results=2))

    if allow_arxiv:
        arxiv_tool = ArxivQueryRun(
            api_wrapper=ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500),
            description="Search academic papers on arXiv"
        )
        tools.append(arxiv_tool)

    # Step 3: Create the ReAct agent with LangGraph
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )

    # Step 4: Setup initial state
    state = {"messages": query}

    # Step 5: Run agent and get response
    response = agent.invoke(state)
    messages = response.get("messages", [])

    # Step 6: Extract the final assistant message
    ai_messages = [m.content for m in messages if isinstance(m, AIMessage)]
    final_message = ai_messages[-1] if ai_messages else "No response from the agent."

    # Step 7: Generate PDF if requested
    pdf_path = generate_pdf_report(final_message) if allow_pdf else None

    # Step 8: Return AI output and optional PDF
    return {
        "response": final_message.strip(),
        "pdf_path": pdf_path
    }
