import asyncio
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
- fact: important info to remember (names, settings, user details)
- instruction: rules the assistant must follow
- decision: a conclusion or choice made
- question: user asking something
- answer: assistant answering something
- chatter: small talk, greetings, filler like "ok", "thanks", "sounds good"

Return ONLY a JSON array of types in the same order.
Example: ["fact", "chatter", "question", "answer"]""",
        messages=[{"role": "user", "content": numbered}]
    )

    raw = response.content[0].text.strip()

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
    )

    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        system="""Summarize this conversation into 2-3 sentences.
Keep all important facts, decisions, and instructions.
Drop all chatter and filler.
Write in third person: 'The user said... The assistant explained...'
Return ONLY the summary, no extra text.""",
        messages=[{"role": "user", "content": conversation}]
    )

    return response.content[0].text.strip()