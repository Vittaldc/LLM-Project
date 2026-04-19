import streamlit as st
import chromadb
import ollama
import sqlite3
import json
import re
import os

# ─────────────────────────────────────────────────────────────
# CONFIG — reads best model from benchmark result if available
# ─────────────────────────────────────────────────────────────
CONFIG_FILE  = "./config.json"
CHROMA_PATH  = "./chroma_db"
DB_PATH      = "./quiz_scores.db"
EMBED_MODEL  = "nomic-embed-text"
MODELS       = ["gemma2:9b", "llama3:8b"]
MODEL_LABELS = {"gemma2:9b": "Gemma 2 9B", "llama3:8b": "Llama 3 8B"}

if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        _cfg = json.load(f)
    DEFAULT_MODEL = _cfg.get("best_model", "gemma2:9b")
else:
    DEFAULT_MODEL = "gemma2:9b"


# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            session   TEXT,
            score     INTEGER,
            total     INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────
# CHROMADB — cached so it loads once per session
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection("economy")


def retrieve_context(query, n=8):
    col    = get_chroma_collection()
    q_emb  = ollama.embed(model=EMBED_MODEL, input=query)["embeddings"][0]
    result = col.query(query_embeddings=[q_emb], n_results=n)
    chunks  = result["documents"][0]
    sources = [m["source"] for m in result["metadatas"][0]]
    context = ""
    for chunk, src in zip(chunks, sources):
        label    = "Ramesh Singh" if "economy_book" in src.lower() else "Economic Survey"
        context += f"\n[Source: {label}]\n{chunk}\n"
    return context


# ─────────────────────────────────────────────────────────────
# QUESTION TYPE DETECTION
# ─────────────────────────────────────────────────────────────
def detect_question_type(question):
    """
    Returns 'factual' for one-line questions (year, name, full-form etc.)
    Returns 'explanation' for concept / analysis questions.
    """
    q = question.lower().strip()

    one_line_triggers = [
        "when was", "in which year", "who is", "who was",
        "what year", "what is the full form", "what does",
        "stand for", "founded in", "established in",
        "headquarters of", "current", "latest", "how many",
        "which article", "which act", "name the", "full form",
        "abbreviation", "expand", "what is the capital",
        "who heads", "who chairs",
    ]
    explain_triggers = [
        "explain", "describe", "elaborate", "discuss",
        "what is the impact", "how does", "why does",
        "what are the effects", "differentiate", "compare",
        "what are the causes", "critically analyse",
        "analyse", "evaluate", "justify", "illustrate",
    ]

    for t in one_line_triggers:
        if t in q:
            return "factual"
    for t in explain_triggers:
        if t in q:
            return "explanation"

    return "explanation"   # safe default


# ─────────────────────────────────────────────────────────────
# SCORING — two separate strategies
# ─────────────────────────────────────────────────────────────
def score_factual(question, answer):
    """
    For one-line / factual questions.
    Rewards brevity, penalises padding.
    """
    score  = 0
    length = len(answer)

    # Length reward — short and direct is best
    if 20 < length < 150:    score += 40   # perfect
    elif length < 250:        score += 25   # acceptable
    elif length > 500:        score -= 20   # penalise padding

    # Key terms from question appear in answer
    q_words   = set(question.lower().split())
    stopwords = {"what","is","the","a","an","of","in","when",
                 "who","which","how","many","was","were","are"}
    key_terms = q_words - stopwords
    matches   = sum(1 for t in key_terms if t in answer.lower())
    score    += min(matches * 10, 30)

    # Contains a year or specific number (good for factual)
    if re.search(r'\b\d{4}\b', answer):   score += 20   # year
    elif re.search(r'\b\d+\b', answer):   score += 10   # any number

    # No filler preamble
    filler = ["certainly", "of course", "great question",
              "sure", "absolutely", "definitely", "i'd be happy"]
    if not any(f in answer.lower() for f in filler):
        score += 10

    return max(0, min(score, 100))


def score_explanation(question, answer):
    """
    For explanation / analysis questions.
    Rewards detail, structure, examples, domain terms.
    """
    score  = 0
    length = len(answer)

    # Length
    if length > 800:    score += 25
    elif length > 400:  score += 15
    elif length > 200:  score += 8
    elif length < 100:  score -= 10

    # Structure
    if "\n" in answer:            score += 10
    if answer.count("\n") > 3:    score += 5

    # Key terms from question
    q_words   = set(question.lower().split())
    stopwords = {"what","is","the","a","an","of","in","and",
                 "to","how","why","does","are","explain","define",
                 "tell","me","describe"}
    key_terms = q_words - stopwords
    matches   = sum(1 for t in key_terms if t in answer.lower())
    score    += min(matches * 5, 20)

    # Examples / data
    example_indicators = [
        "for example", "for instance", "such as", "e.g",
        "%", "crore", "lakh", "billion", "million", "rs.", "₹",
    ]
    if any(ind in answer.lower() for ind in example_indicators):
        score += 15

    # Economics domain terms
    domain_terms = [
        "rbi", "gdp", "fiscal", "monetary", "inflation",
        "deficit", "surplus", "policy", "india", "economy",
        "growth", "therefore", "however", "whereas", "according",
    ]
    domain_count = sum(1 for d in domain_terms if d in answer.lower())
    score       += min(domain_count * 2, 15)

    # Repetition penalty
    words        = answer.lower().split()
    unique_ratio = len(set(words)) / max(len(words), 1)
    if unique_ratio < 0.4:
        score -= 10

    return max(0, min(score, 100))


def score_response(question, answer):
    """Entry point — detects question type then scores accordingly."""
    q_type = detect_question_type(question)
    if q_type == "factual":
        return score_factual(question, answer)
    return score_explanation(question, answer)


# ─────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────
def build_messages(history, question, context):
    system = """You are an expert teacher on the Indian Economy, trained on
Ramesh Singh's Indian Economy book and the Economic Survey of India.

CRITICAL — match your answer length to the question type:
- Factual questions (year, name, full form, who, when, how many):
  Answer in 1-2 sentences only. Be direct. No padding.
- Explanation questions (explain, describe, discuss, analyse, compare):
  Give a detailed structured answer with examples and data.

Additional rules:
- Answer conversationally and remember chat history
- Mention source (Ramesh Singh or Economic Survey) when relevant
- Never pad short answers with unnecessary context
- Never give one-line answers to explanation questions
- Keep a friendly, teacher-like tone"""

    messages = [{"role": "system", "content": system}]
    for msg in history[-12:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"Context from books:\n{context}\n\nQuestion: {question}"
    })
    return messages


# ─────────────────────────────────────────────────────────────
# STREAM A SINGLE MODEL
# ─────────────────────────────────────────────────────────────
def stream_model(model, messages, placeholder):
    full = ""
    stream = ollama.chat(model=model, messages=messages, stream=True)
    for chunk in stream:
        token  = chunk["message"]["content"]
        full  += token
        placeholder.markdown(full + "▌")
    placeholder.markdown(full)
    return full


# ─────────────────────────────────────────────────────────────
# MCQ GENERATION
# ─────────────────────────────────────────────────────────────
def generate_mcqs(topic="Indian economy"):
    context = retrieve_context(topic, n=12)
    system  = """You are an expert MCQ creator for Indian Economy (UPSC / competitive exams).

Generate exactly 18 high quality MCQs from the provided context.
Rules:
- Questions must be factual and based ONLY on the provided context
- Each question must have 4 options (A, B, C, D)
- Options must be plausible — avoid obviously wrong answers
- Vary difficulty: mix easy, medium and hard questions
- Cover different topics from the context

Return a JSON array ONLY. No preamble, no explanation, no markdown backticks.
Each item must have exactly these fields:
{
  "q": "question text",
  "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
  "answer": "A",
  "justification": "2-3 sentence explanation of why this answer is correct"
}"""
    response = ollama.chat(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Context:\n{context}\n\nGenerate exactly 18 MCQs as a JSON array."}
        ]
    )
    raw   = response["message"]["content"].strip()
    raw   = re.sub(r"```json|```", "", raw).strip()
    match = re.search(r'\[.*\]', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            st.error("Could not parse MCQs. Please try generating again.")
            return []
    return []


# ─────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Economy Bot",
    page_icon="🇮🇳",
    layout="wide"
)

conn = init_db()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🇮🇳 Economy Bot")
    st.caption("Ramesh Singh + Economic Survey · 100% Local")
    st.markdown("---")

    st.markdown("**Answer mode:**")
    show_both = st.toggle("Dual model (both answer)", value=True)
    if show_both:
        st.caption("Both Gemma 2 + Llama 3 answer. Best score wins.")
    else:
        st.caption(f"Only {MODEL_LABELS[DEFAULT_MODEL]} answers. Faster + cooler.")

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("- Factual questions → short direct answer scored")
    st.markdown("- Explanation questions → detailed answer scored")
    st.markdown("- Auto-scorer picks the better model response")
    st.markdown("---")

    if st.button("🗑️ Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption(f"Default model: `{DEFAULT_MODEL}`")
    st.caption(f"Embed model: `{EMBED_MODEL}`")


# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 MCQ Quiz", "📊 My Scores"])


# ═════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ═════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Chat with your Economy Books")
    st.caption("Factual questions get short answers · Explanation questions get detailed answers · Best model wins")

    # Initialise chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": (
                "👋 Hello! I'm your Indian Economy study assistant trained on "
                "**Ramesh Singh's Indian Economy** and the **Economic Survey of India**.\n\n"
                "I automatically detect your question type:\n"
                "- **Factual questions** (year, name, full form) → short direct answer\n"
                "- **Explanation questions** (explain, discuss, analyse) → detailed answer\n\n"
                "For every question I run both **Gemma 2 9B** and **Llama 3 8B** and "
                "give you the better answer automatically. Ask me anything!"
            )
        })

    # Render existing chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask anything about Indian Economy..."):

        # Detect question type upfront and show badge
        q_type = detect_question_type(prompt)

        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            if q_type == "factual":
                st.caption("🎯 Detected: factual question → expecting short direct answer")
            else:
                st.caption("📖 Detected: explanation question → expecting detailed answer")

        # Retrieve context once — shared by both models
        with st.spinner("Retrieving relevant content from books..."):
            context = retrieve_context(prompt)

        history_before = st.session_state.chat_history[:-1]
        messages       = build_messages(history_before, prompt, context)

        responses = {}
        scores    = {}

        if show_both:
            # ── Run both models ──────────────────────────────
            for model in MODELS:
                with st.chat_message("assistant"):
                    st.caption(f"🤖 **{MODEL_LABELS[model]}** is answering...")
                    placeholder = st.empty()
                    response    = stream_model(model, messages, placeholder)
                    responses[model] = response
                    scores[model]    = score_response(prompt, response)

                    # Show score with question type context
                    score_label = (
                        f"Score: `{scores[model]}/100`  "
                        f"({'brevity rewarded' if q_type == 'factual' else 'detail rewarded'})"
                    )
                    st.caption(score_label)

            # ── Pick winner ──────────────────────────────────
            winner        = max(scores, key=scores.get)
            loser         = [m for m in MODELS if m != winner][0]
            winner_label  = MODEL_LABELS[winner]
            loser_label   = MODEL_LABELS[loser]
            winner_resp   = responses[winner]

            with st.chat_message("assistant"):
                st.success(
                    f"🏆 **Best answer: {winner_label}** "
                    f"(scored {scores[winner]}/100 vs "
                    f"{scores[loser]}/100 for {loser_label})"
                )
                st.markdown(winner_resp)
                with st.expander("📚 View source chunks retrieved from books"):
                    st.text(context)

        else:
            # ── Single model mode ────────────────────────────
            with st.chat_message("assistant"):
                st.caption(f"🤖 **{MODEL_LABELS[DEFAULT_MODEL]}** is answering...")
                placeholder    = st.empty()
                winner_resp    = stream_model(DEFAULT_MODEL, messages, placeholder)
                single_score   = score_response(prompt, winner_resp)
                winner_label   = MODEL_LABELS[DEFAULT_MODEL]
                st.caption(
                    f"Score: `{single_score}/100`  "
                    f"({'brevity rewarded' if q_type == 'factual' else 'detail rewarded'})"
                )
                with st.expander("📚 View source chunks retrieved from books"):
                    st.text(context)

        # Save to chat history
        st.session_state.chat_history.append({
            "role":    "assistant",
            "content": f"*(Best answer from {winner_label})*\n\n{winner_resp}"
        })


# ═════════════════════════════════════════════════════════════
# TAB 2 — MCQ QUIZ
# ═════════════════════════════════════════════════════════════
with tab2:
    st.subheader("MCQ Quiz — 18 Questions")
    st.markdown("*Every quiz generates fresh questions. Topic is optional.*")

    col1, col2 = st.columns([3, 1])
    with col1:
        topic_input = st.text_input(
            "Topic (optional):",
            placeholder="e.g. monetary policy, balance of payments, inflation, NITI Aayog"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("Generate Quiz 🎯", type="primary")

    if generate_btn:
        topic = topic_input.strip() if topic_input.strip() else "Indian economy concepts"
        with st.spinner(f"Generating 18 questions on '{topic}'... (1-2 minutes)"):
            mcqs = generate_mcqs(topic)
        if mcqs:
            st.session_state.mcqs        = mcqs
            st.session_state.answers     = {}
            st.session_state.submitted   = False
            st.session_state.quiz_topic  = topic
            st.success(f"✅ {len(mcqs)} questions generated on '{topic}'!")
        else:
            st.error("Failed to generate MCQs. Please try again.")

    if "mcqs" in st.session_state and st.session_state.mcqs:
        mcqs = st.session_state.mcqs
        st.markdown("---")
        st.markdown(f"**Topic: {st.session_state.get('quiz_topic', 'General')}**")

        for i, q in enumerate(mcqs):
            st.markdown(f"**Q{i+1}. {q['q']}**")
            choice = st.radio(
                f"q{i}",
                q["options"],
                key=f"q_{i}",
                index=None,
                label_visibility="collapsed"
            )
            if choice:
                st.session_state.answers[i] = choice[0]   # store A/B/C/D
            st.markdown("")

        st.markdown("---")
        answered = len(st.session_state.answers)
        st.caption(f"Answered: {answered} / {len(mcqs)}")

        if st.button("Submit Quiz ✅", type="primary"):
            st.session_state.submitted = True
            score = sum(
                1 for i, q in enumerate(mcqs)
                if st.session_state.answers.get(i, "") == q["answer"]
            )
            st.session_state.score = score
            topic = st.session_state.get("quiz_topic", "general")
            conn.execute(
                "INSERT INTO scores (session, score, total) VALUES (?, ?, ?)",
                (topic, score, len(mcqs))
            )
            conn.commit()

        if st.session_state.get("submitted"):
            score = st.session_state.score
            total = len(mcqs)
            pct   = round(score / total * 100)

            if pct >= 80:
                st.success(f"### 🏆 Score: {score} / {total} ({pct}%) — Excellent!")
            elif pct >= 60:
                st.warning(f"### 👍 Score: {score} / {total} ({pct}%) — Good, keep revising!")
            else:
                st.error(f"### 📖 Score: {score} / {total} ({pct}%) — Need more practice!")

            st.markdown("---")
            st.markdown("### Answer Key with Justifications")
            for i, q in enumerate(mcqs):
                user_ans = st.session_state.answers.get(i, "Not answered")
                correct  = q["answer"]
                if user_ans == correct:
                    st.success(f"✅ Q{i+1}: Your answer **{user_ans}** is correct!")
                else:
                    st.error(
                        f"❌ Q{i+1}: You answered **{user_ans}** — "
                        f"Correct answer is **{correct}**"
                    )
                st.info(f"**Justification:** {q['justification']}")
                st.markdown("")


# ═════════════════════════════════════════════════════════════
# TAB 3 — SCORES
# ═════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Your Quiz History")
    rows = conn.execute(
        "SELECT session, score, total, timestamp "
        "FROM scores ORDER BY timestamp DESC LIMIT 30"
    ).fetchall()

    if rows:
        total_attempts = len(rows)
        avg_pct        = round(sum(r[1]/r[2]*100 for r in rows) / total_attempts)
        best           = max(rows, key=lambda r: r[1]/r[2])

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Attempts", total_attempts)
        m2.metric("Average Score",  f"{avg_pct}%")
        m3.metric("Best Score",     f"{round(best[1]/best[2]*100)}%  ({best[0]})")

        st.markdown("---")
        for r in rows:
            pct = round(r[1]/r[2]*100)
            bar = "🟩" * (pct//10) + "⬜" * (10 - pct//10)
            st.markdown(
                f"{bar} **{pct}%** — {r[1]}/{r[2]} — `{r[0]}` — *{r[3]}*"
            )
    else:
        st.info("No quiz attempts yet. Go to the MCQ Quiz tab to start!")