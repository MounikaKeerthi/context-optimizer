import asyncio
from session import count_tokens_estimate
from agents import classifier_agent, summarizer_agent

def eviction_agent(
    messages: list[dict],
    token_budget: int
) -> tuple[list[dict], str]:

    current_tokens = count_tokens_estimate(messages)
    original_tokens = current_tokens

    result = list(messages)

    for msg in list(result):
        if current_tokens <= int(token_budget * 0.5):
            break
        if msg.get("type") == "chatter":
            result.remove(msg)
            current_tokens -= int(len(msg["content"]) / 4)

    tokens_saved = original_tokens - current_tokens
    action = f"Removed chatter and saved {tokens_saved} tokens."

    return result, action

async def optimize_conversation(
    messages: list[dict],
    token_budget: int
) -> tuple[list[dict], str]:

    classified, summary = await asyncio.gather(
        classifier_agent(messages),
        summarizer_agent(messages[:len(messages)//2])
    )

    evicted, action = eviction_agent(classified, token_budget)

    tokens_after_eviction = count_tokens_estimate(evicted)

    if tokens_after_eviction > int(token_budget * 0.7):
        newer_half = evicted[len(evicted)//2:]
        summary_message = {
            "role": "system",
            "content": f"[CONVERSATION SUMMARY]: {summary}",
            "type": "fact"
        }
        evicted = [summary_message] + newer_half
        action = "Removed chatter and summarized old messages."

    tokens_before = count_tokens_estimate(messages)
    tokens_after = count_tokens_estimate(evicted)

    optimization_message = (
        f"⚡ Auto-optimized: You were at "
        f"{round(tokens_before/token_budget*100)}% capacity. "
        f"After cleanup you are now at "
        f"{round(tokens_after/token_budget*100)}% capacity. "
        f"Freed up {tokens_before - tokens_after} tokens."
    )

    return evicted, optimization_message