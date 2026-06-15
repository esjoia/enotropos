"""enotropos — Streamlit web application.

Chat interface for the wine DOP/IGP RAG assistant.
"""
import streamlit as st

st.set_page_config(
    page_title="enotropos — Wine DOP/IGP Assistant",
    page_icon="🍷",
    layout="wide",
)

st.title("🍷 enotropos")
st.caption("AI assistant for Spanish wine denominations (DOP/IGP)")

# ---- Sidebar ----
with st.sidebar:
    st.header("Filters")
    country = st.selectbox("Country", ["Espanya"], index=0)
    gi_type = st.selectbox("Type", ["All", "DOP", "IGP"], index=0)
    top_k = st.slider("Chunks to retrieve", 3, 10, 5)

    st.divider()
    st.caption(
        "Powered by pymupdf4llm • ChromaDB • DeepSeek V4 Flash\n\n"
        "Answers are based on official EU product specifications."
    )

# ---- Chat ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("Sources"):
                for cit in msg["citations"]:
                    st.markdown(
                        f"- **[{cit['ref']}] {cit['gi_name']}** ({cit['gi_type']}) — "
                        f"{cit['section']} — `{cit['source_file']}`"
                    )

if prompt := st.chat_input("Ask about a wine denomination..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            from winegpt.rag import query_rag

            result = query_rag(
                query=prompt,
                country=country,
                gi_type=None if gi_type == "All" else gi_type,
                top_k=top_k,
            )

        st.markdown(result["answer"])

        if result["citations"]:
            with st.expander("Sources"):
                for cit in result["citations"]:
                    st.markdown(
                        f"- **[{cit['ref']}] {cit['gi_name']}** ({cit['gi_type']}) — "
                        f"{cit['section']} — `{cit['source_file']}`"
                    )

    msg_record = {
        "role": "assistant",
        "content": result["answer"],
        "citations": result["citations"],
    }
    st.session_state.messages.append(msg_record)
