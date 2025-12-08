# app_streamlit.py

import streamlit as st
from rag_core import init_rag, get_answer, get_environment

st.set_page_config(
    page_title="Game NPC RAG Demo",
    layout="wide",
)

st.title("🎮 Game Knowledge NPC - RAG Demo")

st.markdown(
    "Ask a question about the games in our dataset. "
    "This app runs a **RAG pipeline** (retrieval + generation) and records performance metrics."
)

@st.cache_resource
def load_rag():
    init_rag()    # heavy stuff happens only on the first call
    return True

# initialize RAG components once
if "rag_initialized" not in st.session_state:
    with st.spinner("Loading model and index..."):
        load_rag()
    st.session_state["rag_initialized"] = True
    st.session_state["metrics_log"] = []
    st.success("RAG initialized")

# runtime info display
environment = get_environment()
env_col = st.columns(1)[0]
with env_col:
    st.write(f"**Environment:** `{environment}`")

st.markdown("---")

# config outputs
col1, col2 = st.columns(2)
with col1:
    top_k = st.number_input(
        "Top-k retrieved passages",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
    )
with col2:
    max_new_tokens = st.number_input(
        "Max new tokens",
        min_value=32,
        max_value=512,
        value=256,
        step=32,
    )

question = st.text_area(
    "Your question:",
    placeholder="e.g., What is a witcher?",
    height=100,
)

run_clicked = st.button("Ask librarian NPC")

if run_clicked and question.strip():
    with st.spinner("Loading..."):
        result = get_answer(
            question=question.strip(),
            max_new_tokens=int(max_new_tokens),
            top_k=int(top_k),
        )

    # query output
    st.subheader("NPC Answer")
    st.markdown(result["answer"])

    st.subheader("Current Query Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Total time (s)", f"{result['total_time']:.2f}")
    with m2:
        st.metric("Retrieval time (s)", f"{result['retrieval_time']:.2f}")
    with m3:
        st.metric("Generation time (s)", f"{result['generation_time']:.2f}")
    with m4:
        st.metric("Top-k", int(top_k))
    with m5:
        st.metric("Input tokens", result["input_tokens"])
    with m6:
        st.metric("Output tokens", result["output_tokens"])

    # log all metrics
    log_entry = {
        "question": question.strip(),
        "total_time": result["total_time"],
        "retrieval_time": result["retrieval_time"],
        "generation_time": result["generation_time"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "top_k": int(top_k),
        "max_new_tokens": int(max_new_tokens),
        "environment": result["environment"],
    }
    st.session_state["metrics_log"].append(log_entry)

    # retrieved context display
    with st.expander("Retrieved context (evidence)"):
        for i, doc in enumerate(result["retrieved_docs"]):
            st.markdown(f"**Document {i+1}**")
            st.write(doc.page_content)
            if hasattr(doc, "metadata") and doc.metadata:
                st.caption(str(doc.metadata))
            st.markdown("---")

elif run_clicked:
    st.warning("Please enter a question first.")

# metrics history and log
if st.session_state.get("metrics_log"):
    st.markdown("---")
    st.subheader("Aggregate Metrics Over Session")

    log = st.session_state["metrics_log"]
    n = len(log)

    avg_total = sum(x["total_time"] for x in log) / n
    avg_retrieval = sum(x["retrieval_time"] for x in log) / n
    avg_generation = sum(x["generation_time"] for x in log) / n
    avg_in_tokens = sum(x["input_tokens"] for x in log) / n
    avg_out_tokens = sum(x["output_tokens"] for x in log) / n

    a1, a2, a3, a4, a5 = st.columns(5)
    with a1:
        st.metric("Avg total time (s)", f"{avg_total:.2f}")
    with a2:
        st.metric("Avg retrieval time (s)", f"{avg_retrieval:.2f}")
    with a3:
        st.metric("Avg generation time (s)", f"{avg_generation:.2f}")
    with a4:
        st.metric("Avg input tokens", f"{avg_in_tokens:.1f}")
    with a5:
        st.metric("Avg output tokens", f"{avg_out_tokens:.1f}")

    # raw log as a table
    with st.expander("Full metrics log"):
        import pandas as pd
        df = pd.DataFrame(log)
        st.dataframe(df)
