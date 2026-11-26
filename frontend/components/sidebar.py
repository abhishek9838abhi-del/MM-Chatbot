"""Sidebar component for conversation history, document knowledge base, and assistant settings."""

import streamlit as st

from frontend.utils import api_client


def render_sidebar() -> None:
    """Render sidebar controls for conversations, RAG documents, and voice preferences."""
    with st.sidebar:
        st.markdown("### Multimodal AI")
        st.caption("Text · Vision · Audio · RAG")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("+ New Chat", use_container_width=True, type="primary"):
                try:
                    conv = api_client.create_conversation("New Conversation")
                    st.session_state.conversation_id = conv["id"]
                    st.session_state.messages = []
                    st.rerun()
                except api_client.BackendAPIError as exc:
                    st.error(str(exc))

        with col2:
            active_id = st.session_state.get("conversation_id")
            if st.button("Clear Chat", use_container_width=True, disabled=active_id is None):
                try:
                    api_client.clear_conversation_messages(active_id)
                    st.session_state.messages = []
                    st.toast("Conversation messages cleared.")
                    st.rerun()
                except api_client.BackendAPIError as exc:
                    st.error(str(exc))

        st.divider()
        st.markdown("#### Preferences")
        st.session_state.use_rag = st.toggle(
            "Use Document Knowledge (RAG)",
            value=st.session_state.get("use_rag", True),
        )
        st.session_state.generate_audio = st.toggle(
            "Voice Responses (TTS)",
            value=st.session_state.get("generate_audio", False),
        )

        st.divider()
        st.markdown("#### Conversations")
        try:
            conversations = api_client.list_conversations()
        except api_client.BackendAPIError as exc:
            st.warning(str(exc))
            conversations = []

        if not conversations:
            st.caption("No saved conversations yet.")
        else:
            for conv in conversations:
                is_active = conv["id"] == st.session_state.get("conversation_id")
                c_btn, c_del = st.columns([5, 1])
                with c_btn:
                    label = f"● {conv['title']}" if is_active else conv["title"]
                    if st.button(label, key=f"conv_{conv['id']}", use_container_width=True):
                        try:
                            detail = api_client.get_conversation(conv["id"])
                            st.session_state.conversation_id = detail["id"]
                            st.session_state.messages = detail.get("messages", [])
                            st.rerun()
                        except api_client.BackendAPIError as exc:
                            st.error(str(exc))
                with c_del:
                    if st.button("✕", key=f"del_conv_{conv['id']}", help="Delete conversation"):
                        try:
                            api_client.delete_conversation(conv["id"])
                            if is_active:
                                st.session_state.conversation_id = None
                                st.session_state.messages = []
                            st.rerun()
                        except api_client.BackendAPIError as exc:
                            st.error(str(exc))

        st.divider()
        st.markdown("#### Knowledge Base (PDF / TXT)")
        doc_file = st.file_uploader(
            "Upload document for RAG",
            type=["pdf", "txt"],
            key="sidebar_doc_uploader",
            label_visibility="collapsed",
        )
        if doc_file is not None and st.button("Index Document", use_container_width=True):
            with st.spinner("Extracting, chunking & indexing..."):
                try:
                    res = api_client.upload_document(
                        file_bytes=doc_file.getvalue(),
                        filename=doc_file.name,
                        conversation_id=st.session_state.get("conversation_id"),
                    )
                    st.success(res["message"])
                    st.rerun()
                except api_client.BackendAPIError as exc:
                    st.error(str(exc))

        try:
            docs = api_client.list_documents(st.session_state.get("conversation_id"))
        except api_client.BackendAPIError:
            docs = []

        for doc in docs:
            d_col1, d_col2 = st.columns([5, 1])
            with d_col1:
                st.caption(f"📄 {doc['filename']} ({doc['chunk_count']} chunks)")
            with d_col2:
                if st.button("✕", key=f"del_doc_{doc['id']}", help="Remove document"):
                    try:
                        api_client.delete_document(doc["id"])
                        st.rerun()
                    except api_client.BackendAPIError as exc:
                        st.error(str(exc))
