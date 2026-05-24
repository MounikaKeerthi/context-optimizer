from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
import os
from dotenv import load_dotenv
from session import get_session, save_session, count_tokens_estimate, client
from optimizer import optimize_conversation

load_dotenv()

app = FastAPI(
    title="Context Optimizer",
    description="Auto-optimizes chatbot conversations to never hit token limits"
)

class ChatRequest(BaseModel):
    session_id: str
    message: str
    token_budget: int = 2000

class ChatResponse(BaseModel):
    response: str
    tokens_used: int
    token_budget: int
    usage_percent: float
    was_optimized: bool
    optimization_message: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    session = get_session(request.session_id, request.token_budget)
    messages = session["messages"]
    token_budget = session["token_budget"]

    messages.append({"role": "user", "content": request.message, "type": "unknown"})

    claude_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m["role"] in ["user", "assistant"]
    ]

    remaining_tokens = token_budget - count_tokens_estimate(messages)
    max_response_tokens = max(50, min(150, remaining_tokens))

    try:
        claude_response = await asyncio.to_thread(
            client.messages.create,
            model="claude-haiku-4-5-20251001",
            max_tokens=max_response_tokens,
            system="You are a helpful assistant. Keep responses concise, 1-2 sentences max. No bullet points, no markdown, no emojis.",
            messages=claude_messages
        )
        response_text = claude_response.content[0].text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    messages.append({
        "role": "assistant",
        "content": response_text,
        "type": "answer"
    })

    session["messages"] = messages
    save_session(request.session_id, session)

    tokens_used = count_tokens_estimate(messages)
    usage_percent = round(min(tokens_used / token_budget * 100, 100), 1)

    was_optimized = False
    optimization_message = None

    if usage_percent >= 70:
        messages, optimization_message = await optimize_conversation(messages, token_budget)
        save_session(request.session_id, {
            "messages": messages,
            "token_budget": token_budget
        })
        tokens_used = count_tokens_estimate(messages)
        usage_percent = round(min(tokens_used / token_budget * 100, 100), 1)
        was_optimized = True

    return ChatResponse(
        response=response_text,
        tokens_used=tokens_used,
        token_budget=token_budget,
        usage_percent=usage_percent,
        was_optimized=was_optimized,
        optimization_message=optimization_message
    )