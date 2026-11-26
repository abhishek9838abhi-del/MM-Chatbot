# Multimodal AI Chatbot

A full-stack, modular **Multimodal AI Chatbot** built with **FastAPI**, **Streamlit**, **MySQL (SQLAlchemy ORM)**, **Hugging Face Inference Models**, and **FAISS RAG**.

## Features

- **Text Chat**: Context-aware conversations with persistent MySQL history.
- **Image Understanding**: Upload `JPG`, `JPEG`, `PNG`, or `WEBP` images and ask questions using Vision-Language models (`Qwen/Qwen2.5-VL-7B-Instruct`).
- **Audio Input & Output**: Record via microphone or upload audio (`WAV`, `MP3`, `OGG`, `M4A`, `FLAC`, `WEBM`) for speech-to-text (`openai/whisper-large-v3-turbo`), plus voice playback via Hugging Face TTS (`facebook/mms-tts-eng`) with local `pyttsx3` / `gTTS` fallback.
- **Document Q&A (RAG)**: Upload `PDF` and `TXT` files, extract and chunk text, generate Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`), index vectors with FAISS, and answer questions with source chunk citations.
- **Personal Emotional Support & Safety Guardrails**: Empathetic non-judgmental support, self-harm crisis detection with helpline resources, prompt-injection neutralization on retrieved documents, strict upload validation, and rate limiting.

## Project Structure

```text
multimodal_chatbot/
├── frontend/
│   ├── app.py
│   ├── components/
│   │   ├── chat_view.py
│   │   └── sidebar.py
│   ├── styles/
│   │   └── custom.css
│   └── utils/
│       └── api_client.py
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── routes/
│   │   ├── chat.py
│   │   ├── image.py
│   │   ├── audio.py
│   │   └── documents.py
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── vision_service.py
│   │   ├── speech_service.py
│   │   └── rag_service.py
│   ├── database/
│   │   ├── connection.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   └── schema.sql
│   └── security/
│       └── guardrails.py
├── storage/
│   ├── documents/
│   ├── images/
│   ├── audio/
│   └── vectors/
├── tests/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup & Installation

### 1. Create Virtual Environment & Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure MySQL Database

Ensure MySQL Server is running, then run the schema script:

```sql
SOURCE backend/database/schema.sql;
```

Tables created automatically (or via `schema.sql`):
- `conversations`
- `messages`
- `uploaded_documents`
- `document_chunks`

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and set your credentials:

```powershell
copy .env.example .env
```

Set at minimum:
- `HF_TOKEN`: Your Hugging Face API token.
- `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`: Your MySQL connection details.

## Running the Application

Start the **FastAPI Backend**:

```powershell
python -m uvicorn backend.main:app --reload
```

Start the **Streamlit Frontend** (in a second terminal):

```powershell
python -m streamlit run frontend/app.py
```

> **Note (Windows Application Control):** If `.venv\Scripts\python.exe` encounters `DLL load failed ... An Application Control policy has blocked this file` for C-extension wheels (such as NumPy or SQLAlchemy), run with your trusted system `python` interpreter as shown above.

- Frontend UI: `http://localhost:8501`
- Backend API Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

## Running Tests

```powershell
.\.venv\Scripts\pytest.exe -v
```
