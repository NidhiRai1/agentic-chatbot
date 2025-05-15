# .env loading (optional)
from dotenv import load_dotenv
load_dotenv()

# FastAPI setup
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from collections import deque, defaultdict
import difflib

from ai_agent import get_response_from_ai_agent
from faq_data import faq  # 🔥 Import your FAQ dict

# Session-based in-memory history
session_histories = defaultdict(lambda: deque(maxlen=10))

# Allowed model names
ALLOWED_MODEL_NAMES = ["llama3-70b-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile", "gpt-4o-mini"]

# FastAPI instance
app = FastAPI(title="LangGraph AI Agent")

# Request schema
class RequestState(BaseModel):
    session_id: str
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

# Optional fuzzy FAQ match
def check_faq_match(user_query: str):
    closest = difflib.get_close_matches(user_query, faq.keys(), n=1, cutoff=0.8)
    if closest:
        return faq[closest[0]]
    return None

# Chat endpoint with FAQ + history + AI
@app.post("/chat")
def chat_endpoint(request: RequestState):
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model"}

    session_id = request.session_id
    user_input = request.messages[-1]  # Get latest message only
    history = session_histories[session_id]

    # 1. FAQ check
    faq_answer = check_faq_match(user_input)
    if faq_answer:
        history.append(f"User: {user_input}")
        history.append(f"AI: {faq_answer}")
        return {"response": faq_answer, "source": "faq"}

    # 2. Use last 10 message pairs (flattened) for context
    past_context = list(history)
    full_query = past_context + [f"User: {user_input}"]

    # 3. Get AI response
    ai_response = get_response_from_ai_agent(
        llm_id=request.model_name,
        query=full_query,
        allow_search=request.allow_search,
        system_prompt=request.system_prompt,
        provider=request.model_provider
    )

    # 4. Store in history
    history.append(f"User: {user_input}")
    history.append(f"AI: {ai_response['response'] if isinstance(ai_response, dict) else ai_response}")

    return {"response": ai_response, "source": "ai"}

# Run app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
