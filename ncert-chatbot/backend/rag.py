from dotenv import load_dotenv
load_dotenv(override=True)

import os
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from .embed import get_embeddings

DECLINE = ("Sorry, I can only help with topics covered in the NCERT Class 10 Science textbook, "
           "and I couldn't find this in it. Could you ask something from the book?")

SYSTEM = ("You are a friendly tutor for NCERT Class 10 Science. Answer the student's question using the "
          "CONTEXT from the textbook below. The context is always relevant to the question — use it. "
          "Only if the context is completely unrelated to the question, reply with exactly OUT_OF_SCOPE. "
          "For numericals, show formula, substitution and result with units. "
          "Never mention the word 'context'.")


class Bot:
    def __init__(self):
        self.emb = get_embeddings()
        self.vs = FAISS.load_local("index", self.emb, allow_dangerous_deserialization=True)
        self.llm = ChatOpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            model=os.environ["LLM_MODEL"],
            temperature=0,
        )

    def standalone(self, history, msg):
        h = "\n".join(f"{r}: {t}" for r, t in history[-6:])
        p = (f"Rewrite the last student message as one complete, self-contained question. "
             f"Output only the question.\n\n{h}\nstudent: {msg}")
        return self.llm.invoke(p).content.strip()

    def answer(self, q, history=None):
        """Returns (reply, citations, answered). Declines are never cached."""
        res = self.vs.similarity_search_with_score(q, k=4)

        # Scope gate: too weak a match -> decline without calling the LLM.
        if not res or 1 - res[0][1] / 2 < 0.25:
            return DECLINE, [], False

        ctx = "\n\n".join(f"[{d.metadata['chapter']}]\n{d.page_content}" for d, _ in res)
        h = "".join(f"{r}: {t}\n" for r, t in (history or [])[-4:])

        out = self.llm.invoke([
            ("system", SYSTEM),
            ("human", f"CONTEXT:\n{ctx}\n\nCHAT SO FAR:\n{h}\nQUESTION: {q}"),
        ]).content.strip()

        # NOTE: removed the OUT_OF_SCOPE short-circuit — the vector score gate above
        # already handles out-of-scope questions, and the extra check made the model
        # decline perfectly good in-scope questions.

        cites = list(dict.fromkeys(d.metadata["chapter"] for d, _ in res))[:2]
        return out, cites, True