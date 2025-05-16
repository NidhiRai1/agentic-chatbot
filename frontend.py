import streamlit as st
import requests
from PIL import Image
from pytesseract import image_to_string
import io

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("🤖 AI Chatbot Agent")

# --- Agent Configuration ---
with st.expander("⚙️ Agent Configuration", expanded=True):
    system_prompt = st.text_area("System Prompt:", "You are a helpful assistant.")
    provider = st.radio("Select Provider:", ["Groq", "OpenAI"])

    MODEL_NAMES = {
        "Groq": ["llama3-70b-8192", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        "OpenAI": ["gpt-4o-mini"]
    }
    model = st.selectbox("Select Model:", MODEL_NAMES[provider])
    session_id = st.text_input("Session ID:", value="user_001")

    allow_web_search = st.checkbox("Allow Web Search")
    allow_arxiv = st.checkbox("Enable arXiv Search")
    allow_pdf = st.checkbox("Generate PDF Report")

# --- Unified Chat Input ---
st.subheader("💬 Ask a Question (Text / Image / Both)")

with st.form("chat_form"):
    user_question = st.text_area("Your message (optional):", height=100)
    uploaded_image = st.file_uploader("Upload an image (optional):", type=["png", "jpg", "jpeg"])
    submit = st.form_submit_button("Send")

# --- Handle Submission ---
if submit:
    if not user_question and not uploaded_image:
        st.warning("Please enter a message or upload an image.")
    else:
        # Show image and OCR result (optional)
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            extracted_text = image_to_string(image)
            st.write("📝 Extracted Text from Image:")
            st.code(extracted_text)

        # Prepare payload
        data = {
            "system_prompt": system_prompt,
            "provider": provider.lower(),
            "model_name": model,
            "session_id": session_id,
            "user_input": user_question or "",  # ✅ always include even if empty
            "allow_search": str(allow_web_search).lower(),
            "allow_arxiv": str(allow_arxiv).lower(),
            "allow_pdf": str(allow_pdf).lower(),
        }

        files = {"image": uploaded_image.getvalue()} if uploaded_image else None

        try:
            response = requests.post(
                url="http://localhost:9999/chat_with_image_text",
                data=data,
                files=files
            )

            if response.status_code == 200:
                result = response.json()
                st.success("✅ Agent Response:")
                st.markdown(result["response"])

                if result.get("pdf_path"):
                    with open(result["pdf_path"], "rb") as f:
                        st.download_button(
                            label="📄 Download PDF",
                            data=f,
                            file_name=result["pdf_path"].split("/")[-1],
                            mime="application/pdf"
                        )
            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"🔌 Request failed: {e}")
