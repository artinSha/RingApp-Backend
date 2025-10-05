
import os
import json
import random
import sys
from typing import Dict

import google.generativeai as genai
from dotenv import load_dotenv

# -------------------------
# Config / constants
# -------------------------
MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.5-flash-lite")
SCENARIOS_PATH = os.getenv("SCENARIOS_PATH", "scenario/scenario.json")
FORCE_SCENARIO = os.getenv("FORCE_SCENARIO")  # set to a key from scenarios.json to force one
SEED = os.getenv("RANDOM_SEED")

OPENERS = [
    "Let’s jump in. Describe what you see, hear, and feel. Then tell me your first two actions and why.",
    "We’ve got choices. Explain your plan in 3–4 short sentences and the reason behind it.",
    "First, paint the scene in your own words. Then tell me what you do next and what could go wrong.",
]

# -------------------------
# Core helpers
# -------------------------
def configure_genai() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API or GEMINI_API_KEY in your .env")
    genai.configure(api_key=api_key)

def ensure_model_exists(model_id: str) -> None:
    try:
        _ = genai.GenerativeModel(model_id)
    except Exception as e:
        print("Model init raised:", e)
        print("\nAvailable models:")
        for m in genai.list_models():
            print("-", m.name)
        raise

def load_scenarios(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

def build_system_instruction(s: Dict) -> str:
    title = s.get("title", "Scenario")
    setting = s.get("setting", "")
    stakes = s.get("stakes", "")
    role = s.get("role", "")

    return f"""
You are "Oli", an AI calling the learner to simulate a real, unexpected scenario.
Speak ONLY in natural, spoken English. Do NOT write stage directions or placeholders.
Refer to the learner as "you", not by name.

Hard rules (must follow):
- No brackets or parenthetical content: do NOT use (), [], {{}}, <> anywhere.
- No placeholders like [your name], [address], [sound effect], <pause>, *sighs*, etc.
- No scene narration or sound-effect labels (e.g., "(sirens blaring)", "[glass shatters]").
- No markdown formatting or emojis. Plain text only.
- Keep it conversational (1–3 sentences, ≤ 35 words total).
- React to the learner and guide them to describe (avoid yes/no questions).
- Push toward the stakes naturally (why this matters right now).

Scenario Context:
Title: {title}
Setting: {setting}
Goal/Stakes: {stakes}
Role: {role}

Start in the middle of the situation. Speak like you’re on a call. No brackets, no actions, just talk.
""".strip()


def start_session(scenarios: Dict, scenario_key: str):
    """Create model + chat and have the AI speak first. Returns (chat, first_text)."""
    scenario = scenarios[scenario_key]
    system_instruction = build_system_instruction(scenario)

    ensure_model_exists(MODEL_ID)
    model = genai.GenerativeModel(MODEL_ID, system_instruction=system_instruction)
    chat = model.start_chat()

    starter = random.choice(OPENERS)
    first = chat.send_message(starter)
    return chat, first.text

def run_cli_loop(chat, scenario_title: str) -> None:
    print(f"\n🎬 Scenario: {scenario_title}")
    print("(Type 'exit' to end the chat.)\n")
    while True:
        try:
            user_text = input("🧍 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Ending chat. Goodbye!")
            return

        if user_text.lower() == "exit":
            print("👋 Ending chat. Goodbye!")
            return

        try:
            resp = chat.send_message(user_text)
            print(f"🤖 Ring: {resp.text}\n")
        except Exception as e:
            print("Error talking to model:", e)
            print("If it's a 404, set GEMINI_MODEL_ID or change MODEL_ID to one from the list below:\n")
            for m in genai.list_models():
                print("-", m.name)
            return
# __main__
def main():
    if SEED is not None:
        try:
            random.seed(int(SEED))
        except ValueError:
            random.seed(SEED)

    configure_genai()

    # Load scenarios
    try:
        scenarios = load_scenarios(SCENARIOS_PATH)
    except FileNotFoundError:
        print(f"Could not find scenarios at {SCENARIOS_PATH}")
        print("Create a file like:")
        print("""
{
  "zombie_apocalypse": {
    "title": "Zombie Apocalypse",
    "setting": "A small town overrun by the infected. Sirens, empty streets, boarded windows.",
    "stakes": "Stay alive; find safe shelter and supplies.",
    "role": "You and the AI are teammates trying to survive."
  }
}
        """.strip())
        sys.exit(1)

    keys = list(scenarios.keys())
    if not keys:
        print("No scenarios found in the file.")
        sys.exit(1)

    # Pick scenario
    if FORCE_SCENARIO and FORCE_SCENARIO in scenarios:
        scenario_key = FORCE_SCENARIO
    else:
        scenario_key = random.choice(keys)

    scenario = scenarios[scenario_key]
    title = scenario.get("title", scenario_key)
    teaser = scenario.get("setting", "")
    if teaser:
        # Announce in a human, friendly way
        print(f"\n🎲 Random pick: {title}")
        print(f"🗺️  Setting: {teaser}\n")
    else:
        print(f"\n🎲 Random pick: {title}\n")

    # Start chat, AI talks first
    chat, first_text = start_session(scenarios, scenario_key)
    # Explicitly mention the scenario once so learners catch it
    print(f"🤖 Ring: ({title}) {first_text}\n")
    run_cli_loop(chat, title)

if __name__ == "__main__":
    main()
