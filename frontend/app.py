import os, sys, time, threading, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import requests, streamlit as st

# Load LLM env from Streamlit secrets if present (needed only for local mode)
for k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
    try:
        os.environ.setdefault(k, st.secrets[k])
    except Exception:
        pass


@st.cache_resource
def boot():
    """Return the backend URL. Prefer an explicitly-configured remote backend."""
    # 1) Explicit remote backend from Streamlit secrets or environment
    for key in ("API", "API_URL"):
        try:
            v = st.secrets[key]
        except Exception:
            v = os.getenv(key)
        if v:
            return v

    # 2) Otherwise spin up a local in-process backend (for local dev only)
    import uvicorn
    from backend.main import app
    threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning"),
        daemon=True,
    ).start()
    for _ in range(60):
        try:
            requests.get("http://127.0.0.1:8000/docs", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    return "http://127.0.0.1:8000"


API = boot()

st.title("NCERT Class 10 Science Doubt Solver")


def reset():
    st.session_state.sid = requests.post(f"{API}/session").json()["session_id"]
    st.session_state.msgs = []


if "sid" not in st.session_state:
    reset()

st.sidebar.button("New conversation", on_click=reset)

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.write(m["text"])
        if m.get("meta"):
            st.caption(m["meta"])

if prompt := st.chat_input("Ask a doubt from the textbook"):
    st.session_state.msgs.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = requests.post(
                f"{API}/chat",
                json={"session_id": st.session_state.sid, "message": prompt},
                timeout=120,
            )
        if resp.status_code != 200:
            st.error(f"Backend error {resp.status_code}: {resp.text[:500]}")
            st.stop()
        r = resp.json()
        meta = (
            f"Chapter: {', '.join(r['citations']) or '—'} | "
            f"{'⚡ Cache hit' if r['cache_hit'] else 'Fresh answer'} | "
            f"{r['latency_ms']} ms"
        )
        st.write(r["reply"])
        st.caption(meta)
    st.session_state.msgs.append(
        {"role": "assistant", "text": r["reply"], "meta": meta}
    )