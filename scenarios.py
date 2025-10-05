import random
import google.generativeai as genai
from gemini import build_system_instruction, load_scenarios, MODEL_ID, SCENARIOS_PATH

# Load scenarios from JSON
try:
    SCENARIOS = load_scenarios(SCENARIOS_PATH)
except Exception:
    SCENARIOS = {}

_model_cache = {}

def find_scenario_key_by_title(title: str) -> str:
    """
    Map a human-readable scenario title to its internal key in SCENARIOS.
    Falls back to a random one if none match.
    """
    if not title:
        return random.choice(list(SCENARIOS.keys())) if SCENARIOS else "General"
    for key, val in SCENARIOS.items():
        if val.get("title", "").lower() == title.lower():
            return key
    return random.choice(list(SCENARIOS.keys())) if SCENARIOS else "General"


def get_model_for_scenario(scenario_key: str = "General"):
    """
    Return a Gemini model configured with a scenario-specific system instruction.
    """
    if scenario_key in SCENARIOS:
        sys_inst = build_system_instruction(SCENARIOS[scenario_key])
    else:
        sys_inst = (
            "You are 'Ring', a friendly ESL conversation partner. "
            "Reply naturally and briefly, keeping sentences simple and human."
        )

    cache_key = f"{MODEL_ID}::{scenario_key}"
    if cache_key not in _model_cache:
        _model_cache[cache_key] = genai.GenerativeModel(MODEL_ID, system_instruction=sys_inst)
    return _model_cache[cache_key]


def gemini_opening_for_scenario(scenario_key: str) -> str:
    """
    Ask Gemini to produce the first AI line for the selected scenario.
    """
    model = get_model_for_scenario(scenario_key)
    chat = model.start_chat()

    s = SCENARIOS.get(scenario_key, {})
    setting = s.get("setting", "")
    role = s.get("role", "")
    stakes = s.get("stakes", "")

    prompt = (
        "Start the scene now with one short, natural line (<=30 words). "
        "Speak like a human. No explanations — just start the roleplay.\n"
        f"Setting: {setting}\n"
        f"Role: {role}\n"
        f"Stakes: {stakes}"
    )

    try:
        resp = chat.send_message(prompt)
        return (getattr(resp, "text", "") or "").strip() or "Let's begin. What’s happening around you?"
    except Exception as e:
        return f"(Gemini error creating opener: {e})"
    

def get_evaluator_model():
    # A clean model that is NOT role-play; it only returns JSON.
    generation_config = {
        "response_mime_type": "application/json"
    }
    system_inst = (
        "You are an ESL evaluator. You MUST return strictly valid JSON with the exact keys specified. "
        "No prose, no backticks, no extra text—JSON only."
    )
    return genai.GenerativeModel(
        MODEL_ID,  # or a more capable Gemini model if you prefer
        system_instruction=system_inst,
        generation_config=generation_config
    )

