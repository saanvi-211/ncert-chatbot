# NCERT Class 10 Science Chatbot with Smart Caching
Stack: Python, LangChain, FAISS, FastAPI, Streamlit, any OpenAI-compatible LLM (Groq/Gemini free tier).

## Run locally
1. `pip install -r requirements.txt`
2. Download the Class 10 Science chapter PDFs from ncert.nic.in (Textbooks) into `data/`, named after the chapter
   (or add `data/chapters.json`: `{"jesc110": "Light - Reflection and Refraction"}`).
3. `python ingest.py`  (builds `index/`)
4. `cp .env.example .env`, fill in the key, then `export $(cat .env | xargs)`
5. `streamlit run frontend/app.py`  (starts the FastAPI backend in-process automatically)
   or run separately: `uvicorn backend.main:app` and `API_URL=http://127.0.0.1:8000 streamlit run frontend/app.py`

## API
`POST /session` -> `{"session_id"}` · `POST /chat {"session_id","message"}` -> `{reply, citations, cache_hit, latency_ms}`

## Deploy (Streamlit Community Cloud)
Commit `index/`, set `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` in app Secrets, main file `frontend/app.py`.
See EXPLAINER.md for the caching design.
