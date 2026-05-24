import streamlit as st
import requests
import uuid

# Config
API_URL = "https://context-optimizer.onrender.com/chat"
TOKEN_BUDGET = 200

# Page Setup
st.set_page_config(
    page_title="Context Optimizer",
    layout="wide"
)

st.title("Context Optimizer")
st.caption("A chatbot that automatically manages memory to never hit token limits.")

# Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tokens_used" not in st.session_state:
    st.session_state.tokens_used = 0
if "memory" not in st.session_state:
    st.session_state.memory = None
if "optimization_message" not in st.session_state:
    st.session_state.optimization_message = None
if "was_optimized" not in st.session_state:
    st.session_state.was_optimized = False

# Sidebar
with st.sidebar:
    st.header("Memory")
    if st.session_state.memory:
        st.info(st.session_state.memory)
    else:
        st.caption("No memory extracted yet. Keep chatting.")

    st.divider()

    st.header("Session Stats")
    st.metric("Tokens Used", st.session_state.tokens_used)
    st.metric("Token Budget", TOKEN_BUDGET)
    st.metric("Messages", len(st.session_state.messages))

    st.divider()

    if st.button("New Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.tokens_used = 0
        st.session_state.memory = None
        st.session_state.optimization_message = None
        st.session_state.was_optimized = False
        st.rerun()

# Token Usage Bar
usage_percent = st.session_state.tokens_used / TOKEN_BUDGET
st.progress(
    min(usage_percent, 1.0),
    text=f"Token usage: {st.session_state.tokens_used}/{TOKEN_BUDGET} ({round(min(usage_percent * 100, 100), 1)}%)"
)

# Optimization Notification
if st.session_state.was_optimized and st.session_state.optimization_message:
    st.success(st.session_state.optimization_message)
    st.session_state.was_optimized = False

# Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and "api_response" in message:
            with st.expander("View API Response"):
                st.json(message["api_response"])

# Chat Input
if prompt := st.chat_input("Type a message..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Thinking..."):
        try:
            response = requests.post(API_URL, json={
                "session_id": st.session_state.session_id,
                "message": prompt,
                "token_budget": TOKEN_BUDGET
            })
            data = response.json()

            st.session_state.tokens_used = data["tokens_used"]
            st.session_state.was_optimized = data["was_optimized"]
            st.session_state.optimization_message = data["optimization_message"]

            if data.get("memory"):
                st.session_state.memory = data["memory"]

            st.session_state.messages.append({
                "role": "assistant",
                "content": data["response"],
                "api_response": data
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.rerun()