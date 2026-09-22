"""
llm_test.py — test a small local LLM on your PC before moving to the Pi.

Feeds the model REAL current time/date/weather data each turn (instead
of letting it guess), so it answers grounded questions accurately
while still handling open-ended small talk naturally.
"""

from datetime import datetime

import requests
from llama_cpp import Llama

MODEL_PATH = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
WEATHER_API_URL = "http://localhost:3000/api/weather"


def get_context_facts():
    """Build a short block of real, current facts to ground the model."""
    now = datetime.now()
    facts = [
        f"Current time: {now.strftime('%I:%M %p')}",
        f"Current date: {now.strftime('%A, %B %d, %Y')}",
    ]

    try:
        res = requests.get(WEATHER_API_URL, timeout=3)
        data = res.json()
        if "error" not in data:
            facts.append(
                f"Current weather: {data['temp']}°, {data['description']}, "
                f"in {data['city']}"
            )
        else:
            facts.append("Current weather: unavailable")
    except Exception:
        facts.append("Current weather: unavailable")

    return "\n".join(facts)


def build_system_prompt():
    facts = get_context_facts()
    return (
        "You are a helpful voice assistant embedded in a smart mirror. "
        "Keep every answer to one or two short sentences. Be direct and "
        "conversational, no bullet points, no markdown.\n\n"
        "Use ONLY the facts below when asked about the time, date, or "
        "weather — never guess or invent these. If asked something these "
        "facts don't cover, answer normally from general knowledge.\n\n"
        f"{facts}"
    )


def main():
    print("Loading model... (first load can take a bit)")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=1024,      # a bit more headroom than before
        n_threads=4,
        verbose=False,
    )
    print("Ready. Type a message (or 'quit' to stop).")

    while True:
        user_input = input("> ")
        if user_input.lower() == "quit":
            break

        try:
            # Rebuild the system prompt each turn so time/weather stay current.
            system_prompt = build_system_prompt()

            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                max_tokens=80,
                temperature=0.7,
            )
            reply = response["choices"][0]["message"]["content"].strip()
            print(f"Reply: {reply}")
        except Exception as e:
            print(f"Error generating reply: {e}")


if __name__ == "__main__":
    main()