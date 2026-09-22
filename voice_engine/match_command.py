"""
match_command.py — Step 4: test command matching on its own, with typed
text, before wiring in the microphone. Loads commands.json and matches
whatever you type against the phrase lists.

Some replies are special keywords ("time", "date", "weather") that get
replaced with live data instead of a fixed sentence.
"""

import json
from datetime import datetime

import requests  # pip install requests (add to requirements.txt)

with open("commands.json", "r") as f:
    COMMANDS = json.load(f)["commands"]

WEATHER_API_URL = "http://localhost:3000/api/weather"  # your mirror's backend


def get_dynamic_reply(keyword):
    if keyword == "time":
        return f"It's {datetime.now().strftime('%I:%M %p')}."
    if keyword == "date":
        return f"Today is {datetime.now().strftime('%A, %B %d')}."
    if keyword == "weather":
        try:
            res = requests.get(WEATHER_API_URL, timeout=3)
            data = res.json()
            if "error" in data:
                return "Sorry, I couldn't get the weather right now."
            return f"It's {data['temp']} degrees and {data['description']} in {data['city']}."
        except Exception:
            return "Sorry, I couldn't reach the weather service."
    return keyword  # not a special keyword — just a literal reply


def match_command(text):
    text = text.lower().strip()
    for command in COMMANDS:
        for phrase in command["phrases"]:
            if phrase in text:
                return get_dynamic_reply(command["reply"])
    return "Sorry, I don't have a response for that yet."


def main():
    print("Type a command (or 'quit' to stop):")
    while True:
        text = input("> ")
        if text.lower() == "quit":
            break
        reply = match_command(text)
        print(f"Reply: {reply}")


if __name__ == "__main__":
    main()
