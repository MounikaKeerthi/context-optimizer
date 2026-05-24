import anthropic
import os

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

sessions = {}

def count_tokens_estimate(messages: list[dict]) -> int:
    try:
        response = client.messages.count_tokens(
            model="claude-haiku-4-5-20251001",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in messages
                if m["role"] in ["user", "assistant"]
            ]
        )
        return response.input_tokens
    except:
        return int(sum(len(m["content"]) for m in messages) / 4)

def get_session(session_id: str, token_budget: int) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "token_budget": token_budget
        }
    else:
        sessions[session_id]["token_budget"] = token_budget
    return sessions[session_id]

def save_session(session_id: str, session: dict):
    sessions[session_id] = session