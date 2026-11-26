"""Chat message rendering component with multimodal previews, audio playback, and RAG citations."""

import json

import streamlit as st

from frontend.utils import api_client


def _parse_metadata(raw_meta: str | None) -> dict:
    if not raw_meta:
        return {}
    try:
        data = json.loads(raw_meta)
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def render_messages(messages: list[dict]) -> None:
    """Render all messages in the active conversation."""
    if not messages:
        st.info(
            "Welcome! Ask a question below, or use the **Image**, **Audio**, or **Document** "
            "tabs to analyze images, transcribe voice, or chat with your PDFs."
        )
        return

    for idx, msg in enumerate(messages):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        modality = msg.get("modality", "text")
        attachment_path = msg.get("attachment_path")
        meta = _parse_metadata(msg.get("metadata_json"))

        with st.chat_message(role):
            if modality and modality != "text":
                st.markdown(f"<span class='modality-badge'>{modality}</span>", unsafe_allow_html=True)

            if role == "user" and modality == "image" and attachment_path:
                try:
                    img_bytes = api_client.fetch_media_bytes(attachment_path)
                    st.image(img_bytes, width=320)
                except api_client.BackendAPIError:
                    st.caption("Image preview unavailable.")

            if role == "user" and modality == "audio":
                st.caption("Transcribed speech:")

            st.markdown(content)

            # Audio playback for assistant voice replies or user audio
            audio_url = attachment_path if modality in {"audio", "voice"} else meta.get("audio_url")
            if audio_url:
                try:
                    audio_bytes = api_client.fetch_media_bytes(audio_url)
                    st.audio(audio_bytes)
                except api_client.BackendAPIError:
                    st.caption("Audio playback unavailable.")
            elif role == "assistant":
                msg_key = f"tts_btn_{msg.get('id', idx)}_{idx}"
                if st.button("🔊 Listen", key=msg_key):
                    with st.spinner("Generating speech..."):
                        try:
                            tts_res = api_client.synthesize_tts(content)
                            audio_bytes = api_client.fetch_media_bytes(tts_res["audio_url"])
                            st.audio(audio_bytes)
                        except api_client.BackendAPIError as exc:
                            st.warning(str(exc))

            # Display RAG document citations when present
            sources = meta.get("sources") or msg.get("sources") or []
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} chunks)"):
                    for src in sources:
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<strong>{src.get('filename', 'Document')}</strong> "
                            f"(Chunk #{src.get('chunk_index', 0)} · Relevance {src.get('score', 0):.2f})<br>"
                            f"{src.get('excerpt', '')[:280]}..."
                            f"</div>",
                            unsafe_allow_html=True,
                        )
