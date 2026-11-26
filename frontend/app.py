"""Streamlit frontend entrypoint for the Multimodal AI Chatbot."""

from pathlib import Path
import sys

# Ensure project root is importable when invoked via `streamlit run frontend/app.py`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from frontend.components.chat_view import render_messages
from frontend.components.sidebar import render_sidebar
from frontend.utils import api_client

st.set_page_config(
    page_title="Multimodal AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_custom_css() -> None:
    css_path = ROOT_DIR / "frontend" / "styles" / "custom.css"
    if css_path.is_file():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "use_rag" not in st.session_state:
        st.session_state.use_rag = True
    if "generate_audio" not in st.session_state:
        st.session_state.generate_audio = False


def render_header() -> None:
    try:
        health = api_client.get_health()
        db_status = health.get("database", "unavailable")
        provider = health.get("ai_provider", "gemini").upper()
        key_ok = health.get("api_key_configured", False)
        if db_status == "connected" and key_ok:
            badge_html = f"<span class='status-pill status-ok'>API · MySQL · {provider} Ready</span>"
        else:
            badge_html = (
                f"<span class='status-pill status-warn'>"
                f"DB: {db_status} · {provider} Key: {'Set' if key_ok else 'Missing'}"
                f"</span>"
            )
    except api_client.BackendAPIError:
        badge_html = "<span class='status-pill status-err'>Backend Offline</span>"

    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <p class="app-header-title">Multimodal AI Assistant</p>
                <p class="app-header-sub">Context-aware Text, Vision, Speech, and RAG Document Q&amp;A</p>
            </div>
            <div>{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def refresh_active_conversation(conversation_id: int) -> None:
    detail = api_client.get_conversation(conversation_id)
    st.session_state.conversation_id = detail["id"]
    st.session_state.messages = detail.get("messages", [])


def main() -> None:
    load_custom_css()
    init_session_state()
    render_sidebar()
    render_header()

    render_messages(st.session_state.messages)

    with st.expander("Multimodal Inputs (Image · Voice / Mic · Document)", expanded=False):
        tab_img, tab_audio, tab_doc = st.tabs(["🖼️ Image", "🎙️ Audio / Mic", "📄 Document"])

        with tab_img:
            img_file = st.file_uploader(
                "Upload an image (JPG, JPEG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
                key="main_img_uploader",
            )
            img_prompt = st.text_input(
                "Question about the image",
                value="What is in this image? Describe it clearly.",
                key="img_prompt_input",
            )
            if st.button("Analyze Image", disabled=img_file is None, type="primary"):
                with st.spinner("Analyzing image with vision-language model..."):
                    try:
                        res = api_client.analyze_image(
                            file_bytes=img_file.getvalue(),
                            filename=img_file.name,
                            prompt=img_prompt,
                            conversation_id=st.session_state.conversation_id,
                            generate_audio=st.session_state.generate_audio,
                        )
                        refresh_active_conversation(res["conversation_id"])
                        st.rerun()
                    except api_client.BackendAPIError as exc:
                        st.error(str(exc))

        with tab_audio:
            mic_audio = st.audio_input("Record with microphone")
            uploaded_audio = st.file_uploader(
                "Or upload an audio file (WAV, MP3, OGG, M4A, FLAC, WEBM)",
                type=["wav", "mp3", "ogg", "m4a", "flac", "webm"],
                key="main_audio_uploader",
            )
            chosen_audio = mic_audio or uploaded_audio
            if st.button("Transcribe & Send Audio", disabled=chosen_audio is None, type="primary"):
                with st.spinner("Transcribing speech and generating response..."):
                    try:
                        filename = getattr(chosen_audio, "name", "recording.wav")
                        res = api_client.send_audio_chat(
                            file_bytes=chosen_audio.getvalue(),
                            filename=filename,
                            conversation_id=st.session_state.conversation_id,
                            use_rag=st.session_state.use_rag,
                            generate_audio=st.session_state.generate_audio,
                        )
                        refresh_active_conversation(res["conversation_id"])
                        st.rerun()
                    except api_client.BackendAPIError as exc:
                        st.error(str(exc))

        with tab_doc:
            doc_file = st.file_uploader(
                "Upload a PDF or TXT document to index for Q&A",
                type=["pdf", "txt"],
                key="main_doc_uploader",
            )
            if st.button("Upload & Index Document", disabled=doc_file is None, type="primary"):
                with st.spinner("Extracting text and generating embeddings..."):
                    try:
                        res = api_client.upload_document(
                            file_bytes=doc_file.getvalue(),
                            filename=doc_file.name,
                            conversation_id=st.session_state.conversation_id,
                        )
                        st.success(res["message"])
                    except api_client.BackendAPIError as exc:
                        st.error(str(exc))

    user_prompt = st.chat_input("Ask a question, upload a document, or share how you're feeling...")
    if user_prompt:
        with st.spinner("Thinking..."):
            try:
                res = api_client.send_text_chat(
                    message=user_prompt,
                    conversation_id=st.session_state.conversation_id,
                    use_rag=st.session_state.use_rag,
                    generate_audio=st.session_state.generate_audio,
                )
                refresh_active_conversation(res["conversation_id"])
                st.rerun()
            except api_client.BackendAPIError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
