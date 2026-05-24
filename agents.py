import asyncio
from curses import raw
import json
from session import client

async def classifier_agent(messages: list[dict]) -> list[dict]:
    numbered = "\n".join(
        f"{i+1}. {m['content']}"
        for i, m in enumerate(messages)
    )

    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system="""Classify each numbered message into exactly one type.

Types:
- fact: important info to remember (names, settings, user details, personal info, goals)
- instruction: rules the assistant must follow
- decision: a conclusion or choice made
- question: user asking something
- answer: assistant answering something
- chatter: pure filler with zero information ("ok", "thanks", "cool", "great", "good!")

Examples:
"I am Mona" → fact
"I work as a software engineer" → fact
"I want to learn Spanish" → fact
"All I know is bien and gracias" → fact
"ok" → chatter
"thanks!" → chatter
"good!" → chatter
"what is RAG?" → question
"we decided to use PostgreSQL" → decision

Return ONLY a JSON array of types in the same order.
Example: ["fact", "chatter", "question", "answer"]""",
        messages=[{"role": "user", "content": numbered}]
    )

    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        labels = json.loads(raw)
    except json.JSONDecodeError:
        labels = ["chatter"] * len(messages)

    valid = ["fact", "instruction", "decision", "question", "answer", "chatter"]

    return [
        {**m, "type": label if label in valid else "chatter"}
        for m, label in zip(messages, labels)
    ]


async def summarizer_agent(messages: list[dict]) -> str:
    conversation = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in messages
        if m["role"] in ["user", "assistant"]
    )

    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        system="""You are a conversation summarizer.
        Write 2-3 plain English sentences summarizing the conversation.
        Start with 'Earlier in this conversation...'
        Focus on what the user said and what was discussed.
        Never return JSON, arrays, or labels. Only plain sentences.""",
        messages=[{"role": "user", "content": f"Summarize this:\n\n{conversation}"}]
    )

    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return raw


async def memory_agent(messages: list[dict]) -> str:
    conversation = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in messages
        if m["role"] in ["user", "assistant"]
    )

    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        system="""Extract key facts about the user from this conversation.
        Write a single short paragraph starting with 'User facts:'
        Include: name, job, goals, preferences, anything personal they shared.
        If no personal facts exist write: 'User facts: No personal information shared yet.'
        Never return JSON or lists. Only plain sentences.""",
        messages=[{"role": "user", "content": f"Extract user facts from:\n\n{conversation}"}]
    )

    return response.content[0].text.strip()