from dotenv import load_dotenv
load_dotenv()

import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Literal, Optional
from collections import deque, defaultdict
import difflib

from ai_agent import get_response_from_ai_agent
from faq_data import faq
from ocr_tool import extract_text_from_image

# Session memory
session_histories = defaultdict(lambda: deque(maxlen=20))

ALLOWED_MODEL_NAMES = [
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile",
    "gpt-4o-mini"
]

app = FastAPI(title="LangGraph AI Agent")

# --- Base Text Chat Endpoint ---
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

    # Check FAQ fallback
    faq_answer = difflib.get_close_matches(user_input, faq.keys(), n=1, cutoff=0.8)
    if faq_answer:
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": faq[faq_answer[0]]})
        return {
            "response": faq[faq_answer[0]],
            "pdf_path": None,
            "history": list(history)
        }

    # Avoid duplicate entries
    if not history or (history[-1]["role"] != "user" or history[-1]["content"] != user_input):
        history.append({"role": "user", "content": user_input})

    formatted_query = [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history]

    ai_response = get_response_from_ai_agent(
        llm_id=request.model_name,
        query=formatted_query,
        allow_search=request.allow_search,
        allow_arxiv=request.allow_arxiv,
        allow_pdf=request.allow_pdf,
        system_prompt=request.system_prompt,
        provider=request.model_provider
    )

    history.append({"role": "assistant", "content": ai_response["response"]})

    return {
        "response": ai_response["response"],
        "pdf_path": ai_response.get("pdf_path"),
        "history": list(history)
    }

# --- Image Only Chat Endpoint (OCR) ---
@app.post("/chat_with_image")
async def chat_with_image(
    session_id: str = Form(...),
    model_name: str = Form(...),
    model_provider: str = Form(...),
    system_prompt: str = Form(...),
    allow_search: bool = Form(...),
    allow_arxiv: bool = Form(...),
    allow_pdf: bool = Form(...),
    user_text: Optional[str] = Form(""),
    image: UploadFile = File(...)
):
    os.makedirs("temp_images", exist_ok=True)
    image_path = f"temp_images/{image.filename}"
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    ocr_text = extract_text_from_image(image_path)

    if user_text.strip() and ocr_text:
        combined_input = f"{user_text.strip()}\n\n[OCR Extracted Text:]\n{ocr_text}"
    elif ocr_text:
        combined_input = f"[Image Input - OCR Extracted Text:]\n{ocr_text}"
    else:
        combined_input = user_text.strip() or "No user input provided."

    formatted_query = [f"User: {combined_input}"]

    ai_response = get_response_from_ai_agent(
        llm_id=model_name,
        query=formatted_query,
        allow_search=allow_search,
        allow_arxiv=allow_arxiv,
        allow_pdf=allow_pdf,
        system_prompt=system_prompt,
        provider=model_provider
    )

    return {
        "response": ai_response["response"],
        "pdf_path": ai_response.get("pdf_path")
    }

# --- Unified Chat Endpoint (Text + Optional Image) ---
@app.post("/chat_with_image_text")
async def chat_with_image_text(
    system_prompt: str = Form(...),
    provider: str = Form(...),
    model_name: str = Form(...),
    session_id: str = Form(...),
    user_input: str = Form(""),  # ✅ Required with default value to prevent 422 error
    allow_search: str = Form("false"),
    allow_arxiv: str = Form("false"),
    allow_pdf: str = Form("false"),
    image: Optional[UploadFile] = File(None),
):
    ocr_text = ""
    if image:
        image_bytes = await image.read()
        with open("temp_image_upload.png", "wb") as f:
            f.write(image_bytes)
        ocr_text = extract_text_from_image("temp_image_upload.png")

    final_prompt = ""
    if user_input.strip():
        final_prompt += user_input.strip() + "\n\n"
    if ocr_text.strip():
        final_prompt += "Text from image:\n" + ocr_text.strip()

    if not final_prompt.strip():
        final_prompt = "No usable input provided."

    formatted_query = [f"User: {final_prompt}"]

    ai_response = get_response_from_ai_agent(
        llm_id=model_name,
        query=formatted_query,
        allow_search=allow_search.lower() == "true",
        allow_arxiv=allow_arxiv.lower() == "true",
        allow_pdf=allow_pdf.lower() == "true",
        system_prompt=system_prompt,
        provider=provider
    )

    return {
        "response": ai_response["response"],
        "pdf_path": ai_response.get("pdf_path")
    }

# --- Uvicorn Dev Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
