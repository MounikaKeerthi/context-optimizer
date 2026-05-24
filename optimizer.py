import asyncio
from session import count_tokens_estimate
from agents import classifier_agent, summarizer_agent, memory_agent

def eviction_agent(
    messages: list[dict],
    token_budget: int
) -> list[dict]:

    current_tokens = count_tokens_estimate(messages)
    result = list(messages)

    eviction_order = ["chatter", "question", "answer", "decision"]

    for evict_type in eviction_order:
        if current_tokens <= int(token_budget * 0.6):
            break
        for msg in list(result):
            if current_tokens <= int(token_budget * 0.6):
                break
            if msg.get("type") == evict_type and len(result) > 1:
                result.remove(msg)
                current_tokens -= int(len(msg["content"]) / 4)

    return result


async def optimize_conversation(
    messages: list[dict],
    token_budget: int
) -> tuple[list[dict], str, str, list[dict]]:

    classified, summary, memory = await asyncio.gather(
        classifier_agent(messages),
        summarizer_agent(messages[:len(messages)//2]),
        memory_agent(messages)
    )

    evicted = eviction_agent(classified, token_budget)

    tokens_after_eviction = count_tokens_estimate(evicted)
    if tokens_after_eviction > int(token_budget * 0.6):
        newer_half = evicted[len(evicted)//2:]
        summary_message = {
            "role": "system",
            "content": summary,
            "type": "fact"
        }
        evicted = [summary_message] + newer_half

    memory_message = {
        "role": "system",
        "content": memory,
        "type": "fact"
    }
    final_messages = [memory_message] + evicted

    tokens_before = count_tokens_estimate(messages)
    tokens_after = count_tokens_estimate(final_messages)

    optimization_message = (
        f"Auto-optimized: You were at "
        f"{round(min(tokens_before/token_budget*100, 100), 1)}% capacity. "
        f"After cleanup you are now at "
        f"{round(min(tokens_after/token_budget*100, 100), 1)}% capacity. "
        f"Freed up {tokens_before - tokens_after} tokens."
    )

    return final_messages, optimization_message, memory, classified