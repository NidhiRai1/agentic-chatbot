import streamlit as st
import requests

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("AI Chatbot Agents")
st.write("Define an AI Agent and Start Chatting!")

# Sidebar / inputs
system_prompt = st.text_area("System Prompt:", value="You are a helpful assistant.")
provider = st.radio("Select Provider:", ("Groq", "OpenAI"))
MODEL_NAMES = {
    "Groq": ["llama3-70b-8192", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
    "OpenAI": ["gpt-4o-mini"]
}
selected_model = st.selectbox("Select Model:", MODEL_NAMES[provider])
allow_web_search = st.checkbox("Allow Web Search")
allow_arxiv = st.checkbox("Enable arXiv Search")
allow_pdf = st.checkbox("Generate PDF Report")

session_id = st.text_input("Session ID", value="user_001")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_area("Your message:", height=100)

API_URL = "http://127.0.0.1:9999/chat"

def format_history(history):
    formatted = ""
    for msg in history:
        role = msg["role"].capitalize()
        content = msg["content"].strip()
        formatted += f"**{role}**: {content}\n\n"
    return formatted

if st.button("Send"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        # Build message list for sending: just the session's chat history + new user message
        messages = st.session_state.chat_history + [{"role": "user", "content": user_input.strip()}]

        payload = {
            "session_id": session_id,
            "model_name": selected_model,
            "model_provider": provider.lower(),
            "system_prompt": system_prompt,
            "messages": messages,
            "allow_search": allow_web_search,
            "allow_arxiv": allow_arxiv,
            "allow_pdf": allow_pdf
        }

        st.text("Sending request...")

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                data = response.json()
                # Update the chat history in session state
                st.session_state.chat_history = data["history"]

                # Show chat history nicely formatted
                st.subheader("Conversation")
                st.markdown(format_history(st.session_state.chat_history))

                # If PDF generated, show download button
                if data.get("pdf_path"):
                    with open(data["pdf_path"], "rb") as f:
                        st.download_button(
                            label="📄 Download PDF",
                            data=f,
                            file_name=data["pdf_path"].split("/")[-1],
                            mime="application/pdf"
                        )
            else:
                st.error(f"Request failed with status code {response.status_code}")
        except Exception as e:
            st.error(f"Request failed: {e}")
