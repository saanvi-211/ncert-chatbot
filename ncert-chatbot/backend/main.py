import time, uuid, threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .cache import SemanticCache, STYLE, ANAPHORA
from .rag import Bot

app = FastAPI(title="NCERT Class 10 Science Chatbot")

sessions: dict[str, list] = {}
_state = {}
_lock = threading.Lock()

MAX_TURNS = 10  # keep the last N (student, tutor) pairs in history


def deps():
    """Lazily construct the Bot + SemanticCache once, thread-safely."""
    with _lock:
        if not _state:
            bot = Bot()
            _state.update(bot=bot, cache=SemanticCache(bot.emb))
    return _state["bot"], _state["cache"]


class ChatIn(BaseModel):
    session_id: str
    message: str


@app.post("/session")
def new_session():
    sid = str(uuid.uuid4())
    sessions[sid] = []
    return {"session_id": sid}


@app.post("/chat")
def chat(body: ChatIn):
    if body.session_id not in sessions:
        raise HTTPException(404, "unknown session")

    t0 = time.perf_counter()
    hist = sessions[body.session_id]
    msg = body.message.strip()
    bot, cache = deps()

    # Classify the turn.
    style = bool(STYLE.search(msg))                                    # e.g. "explain simply"
    dependent = style or (bool(hist) and bool(ANAPHORA.search(msg)))   # tied to this convo -> never read cache

    # Cache is only safe to consult when the query stands on its own.
    hit = None if dependent else cache.lookup(msg)

    if hit:
        # HIT: embedding + guards only, zero LLM calls.
        reply, cites, hit_flag = hit["reply"], hit["citations"], True
    else:
        try:
            # Rewrite follow-ups (including style+anaphora) into a standalone question.
            q = bot.standalone(hist, msg) if (dependent and hist) else msg
            # Pass history to the LLM only when restyling is requested.
            reply, cites, ok = bot.answer(q, hist if style else None)
        except Exception as e:
            raise HTTPException(502, f"{type(e).__name__}: {e}")

        hit_flag = False
        if ok and not style:
            # Store under the standalone question it actually answers.
            cache.add(q, reply, cites)

    # Append this turn, then cap history length (in place, so sessions[sid] stays the same list).
    hist += [("student", msg), ("tutor", reply)]
    if len(hist) > MAX_TURNS * 2:
        del hist[: -MAX_TURNS * 2]

    return {
        "reply": reply,
        "citations": cites,
        "cache_hit": hit_flag,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }