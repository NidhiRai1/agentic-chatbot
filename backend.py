from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal
from collections import deque, defaultdict
import difflib

from ai_agent import get_response_from_ai_agent
from faq_data import faq

# Store recent messages per session, max 20 messages per session
session_histories = defaultdict(lambda: deque(maxlen=20))

ALLOWED_MODEL_NAMES = [
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile",
    "gpt-4o-mini"
]

app = FastAPI(title="LangGraph AI Agent")

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class RequestState(BaseModel):
    session_id: str
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[Message]
    allow_search: bool
    allow_arxiv: bool
    allow_pdf: bool

@app.post("/chat")
def chat_endpoint(request: RequestState):
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model"}

    session_id = request.session_id
    user_input = request.messages[-1].content.strip()
    history = session_histories[session_id]

    # Step 1: Check FAQ fallback
    faq_answer = difflib.get_close_matches(user_input, faq.keys(), n=1, cutoff=0.8)
    if faq_answer:
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": faq[faq_answer[0]]})
        return {
            "response": faq[faq_answer[0]],
            "pdf_path": None,
            "history": list(history)
        }

    # Step 2: Append only new messages (avoid duplication)
    # We assume the last message in request.messages is the new user input,
    # so append it if it's not already last in history
    if not history or (history and (history[-1]["role"] != "user" or history[-1]["content"] != user_input)):
        history.append({"role": "user", "content": user_input})

    # Step 3: Prepare formatted prompt from history
    formatted_query = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history]

    # Step 4: Get AI agent response
    ai_response = get_response_from_ai_agent(
        llm_id=request.model_name,
        query=formatted_query,
        allow_search=request.allow_search,
        allow_arxiv=request.allow_arxiv,
        allow_pdf=request.allow_pdf,
        system_prompt=request.system_prompt,
        provider=request.model_provider
    )

    # Step 5: Append assistant reply to history
    history.append({"role": "assistant", "content": ai_response["response"]})

    return {
        "response": ai_response["response"],
        "pdf_path": ai_response.get("pdf_path"),
        "history": list(history)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
