from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
from bson.objectid import ObjectId
from flask_cors import CORS
import requests
import tempfile
import base64
from werkzeug.utils import secure_filename
import google.generativeai as genai
import subprocess
import json
import random

from m4atowav import convert_m4a_to_wav 
from STT import transcribe_wav
from gemini import (
    configure_genai,
    ensure_model_exists,
    build_system_instruction,
    load_scenarios,
    MODEL_ID,
    SCENARIOS_PATH,
)


# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

from TTS import elevenlabs_tts_get_bytes

# Flask app
app = Flask(__name__)
CORS(app)


# MongoDB connection
client = MongoClient(MONGO_URI)
db = client["ring_app"]
users_collection = db["users"]
conversations_collection = db["conversations"]


# Create a user
@app.route("/create_user", methods=["POST"])
def create_user():
    data = request.json or {}
    user = {
        "username": data.get("username"),
        "email": data.get("email"),
        "dnd_start": data.get("dnd_start", "09:00"),
        "dnd_end": data.get("dnd_end", "17:00"),
        "device_token": data.get("device_token", None),
        "created_at": datetime.now(timezone.utc)  # Changed here
    }
    res = users_collection.insert_one(user)
    return jsonify({"user_id": str(res.inserted_id)}), 201


def _find_scenario_key_by_title(title: str) -> str:
    """
    Map a human-visible title to the canonical scenario key in SCENARIOS.
    Falls back to a random key, then 'General' if needed.
    """
    if not title:
        return random.choice(list(SCENARIOS.keys())) if SCENARIOS else "General"
    for key, val in SCENARIOS.items():
        if val.get("title", "").lower() == title.lower():
            return key
    return random.choice(list(SCENARIOS.keys())) if SCENARIOS else "General"


"""
Call this endpoint from the frontend once the user accepts the call. First message from AI is sent.
"""
@app.route("/start_call", methods=["POST"])
def start_call():
    data = request.json or {}
    user_id = data.get("user_id")
    scenario_title = data.get("scenario", "General")

    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    # Verify user exists
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        return jsonify({"error": "invalid user_id"}), 400

    # Create conversation document
    conv_doc = {
        "user_id": user_id,
        "scenario": scenario_title,
        "conversation": [],  # store AI+user turns
        "timestamp": datetime.now(timezone.utc),  # Changed here
        "grammar_feedback": None
    }
    conv_res = conversations_collection.insert_one(conv_doc)
    conv_id = str(conv_res.inserted_id)

    # -----------------------
    # Always generate first AI line
    # -----------------------
    # Map title -> canonical scenario key, then ask Gemini for an opener
    scenario_key = _find_scenario_key_by_title(scenario_title)
    ai_text = _gemini_opening_for_scenario(scenario_key)

    # Save first AI turn
    conversations_collection.update_one(
        {"_id": ObjectId(conv_id)},
        {"$push": {"conversation": {
            "turn": 0,
            "user_text": None,
            "ai_text": ai_text,
            "created_at": datetime.now(timezone.utc)
        }}}
    )

    # Placeholder for TTS (ElevenLabs) - return None for now
    ai_audio_url = None

    payload = {
        "conversation_id": conv_id,
        "initial_ai_text": ai_text,
        "initial_ai_audio_url": ai_audio_url
    }

    return jsonify(payload), 201


ALLOWED_EXTENSIONS = {"m4a", "wav"}

def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------
# Helper: Transcription of user audio using Google Cloud STT
# -------------------------------
def transcribe_audio_stt(audio_file):
    """
    Accepts a werkzeug FileStorage (uploaded 'audio').
    If .m4a: converts to .wav with convert_m4a_to_wav(), then transcribes with transcribe_wav().
    If .wav: transcribes directly.
    Returns transcript string.
    """
    if not audio_file or not getattr(audio_file, "filename", ""):
        raise RuntimeError("No audio file provided.")

    filename = secure_filename(audio_file.filename)
    if not _allowed_file(filename):
        raise RuntimeError("Unsupported file type. Please upload .m4a or .wav.")

    # Save the uploaded file to a secure temp path
    in_ext = os.path.splitext(filename)[1].lower()  # ".m4a" or ".wav"
    tmp_in_fd, tmp_in_path = tempfile.mkstemp(suffix=in_ext)
    os.close(tmp_in_fd)
    audio_file.save(tmp_in_path)

    wav_path = None
    try:
        if in_ext == ".m4a":
            # Convert to .wav (mono, 16kHz) using your m4atowav.py
            wav_path = convert_m4a_to_wav(tmp_in_path)
        else:
            # Already wav — use the saved path directly
            wav_path = tmp_in_path

        # Transcribe using your STT.py
        transcript = transcribe_wav(wav_path) or ""
        return transcript.strip()

    finally:
        # Cleanup temp files
        try:
            if tmp_in_path and os.path.isfile(tmp_in_path):
                os.remove(tmp_in_path)
        except Exception:
            pass
        if wav_path and wav_path != tmp_in_path:
            try:
                if os.path.isfile(wav_path):
                    os.remove(wav_path)
            except Exception:
                pass


# ------------------------- Gemini AI setup
# -------------------------
configure_genai()  # uses GEMINI_API or GEMINI_API_KEY

try:
    SCENARIOS = load_scenarios(SCENARIOS_PATH)
except Exception:
    SCENARIOS = {}

_model_cache = {}

def _get_model_for_scenario(scenario_key: str = "General"):
    """
    Return a Gemini model configured with a scenario-specific system instruction.
    Cached per (MODEL_ID, scenario_key).
    """
    if scenario_key in SCENARIOS:
        sys_inst = build_system_instruction(SCENARIOS[scenario_key])
    else:
        sys_inst = (
            "You are 'Ring', a friendly, realistic speaking partner for an ESL learner. "
            "Reply in simple, natural English (<= 35 words), react to the user's last message, "
            "and continue the scene briefly with a mix of statements and questions."
        )

    cache_key = f"{MODEL_ID}::{scenario_key}"
    model = _model_cache.get(cache_key)
    if model is None:
        model = genai.GenerativeModel(MODEL_ID, system_instruction=sys_inst)
        _model_cache[cache_key] = model
    return model

def _gemini_opening_for_scenario(scenario_key: str) -> str:
    """
    Ask Gemini to produce the first short line for the chosen scenario.
    """
    model = _get_model_for_scenario(scenario_key)
    chat = model.start_chat()

    s = SCENARIOS.get(scenario_key, {})
    setting = s.get("setting", "")
    role = s.get("role", "")
    prompt = (
        "Start the scene now with one short, natural line (<=30 words). "
        "Speak like a human. Do not explain rules.\n"
        f"Setting: {setting}\n"
        f"Role: {role}"
    )
    try:
        resp = chat.send_message(prompt)
        text = (getattr(resp, "text", "") or "").strip()
        return text or "Okay, let's begin. What's happening around you right now?"
    except Exception as e:
        return f"(Gemini error creating opener: {e})"
    
def generate_ai_text(conversation_context: str, scenario_key: str = "General") -> str:
    """
    Generates the next AI reply from Gemini using a single string prompt.
    This keeps compatibility with your current /process_audio call.
    """
    try:
        model = _get_model_for_scenario(scenario_key)
        chat = model.start_chat()
        resp = chat.send_message(conversation_context or "...")
        ai_text = (getattr(resp, "text", "") or "").strip()
        if not ai_text:
            ai_text = "I couldn’t quite hear that. Could you say it again, briefly?"
        return ai_text
    except Exception as e:
        # Don't crash your request path if Gemini misconfigures
        return f"(Gemini error: {e})"

# -------------------------------
# Endpoint: process user audio
# -------------------------------

@app.route("/process_audio", methods=["POST"])
def process_audio():
    # Get form data
    conv_id = request.form.get("conv_id")
    audio_file = request.files.get("audio")

    if not conv_id or not audio_file:
        return jsonify({"error": "conv_id and audio file are required"}), 400

    # Verify conversation exists
    conversation = conversations_collection.find_one({"_id": ObjectId(conv_id)})
    if not conversation:
        return jsonify({"error": "Invalid conversation ID"}), 400

    #Transcribe audio using speech to text (placeholder)
    user_response = transcribe_audio_stt(audio_file)

    # Fetch the previous turn
    last_turn = conversation["conversation"][-1]
    
    
    if last_turn["ai_text"] and last_turn["user_text"] is None:
        #We know here that nothing wrong has happened with the AI/User turn order...
        conversations_collection.update_one(
            {"_id": ObjectId(conv_id), "conversation.turn": last_turn["turn"]},
            {"$set": {"conversation.$.user_text": user_response}}
        )
    else:
        #Something has gone wrong with the User/AI turn order
        #The most recent turn does not have format AI:'sampletext', User_text:None
        print("User/AI turn order has gone wrong. Check line 170, app.py")
        
    

    #Fetch the updated conversation
    conversation = conversations_collection.find_one({"_id": ObjectId(conv_id)})

    # Build conversation context (including the full history)
    context_text = ""
    for turn_data in conversation.get("conversation", []):
        ai_text = turn_data.get("ai_text")
        user_text = turn_data.get("user_text")

        if ai_text and ai_text.strip():
            context_text += f"AI: {ai_text}\n"

        if user_text and user_text.strip():
            context_text += f"User: {user_text}\n"


    # Generate AI response (placeholder)
    ai_text = generate_ai_text(context_text)

    # Create a new AI-only turn
    new_turn_number = last_turn["turn"] + 1
    conversations_collection.update_one(
        {"_id": ObjectId(conv_id)},
        {"$push": {"conversation": {
            "turn": new_turn_number,
            "user_text": None,
            "ai_text": ai_text,
            "created_at": datetime.now(timezone.utc)
        }}}
    )

    # Generate TTS via ElevenLabs and return base64
    try:
        ai_audio_bytes = elevenlabs_tts_get_bytes(ai_text)
        ai_audio_b64 = base64.b64encode(ai_audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"TTS Error: {str(e)}")  # Log the actual error
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500
        

    # Return response to frontend
    return jsonify({
        "user_text": user_response,
        "ai_text": ai_text,
        "ai_audio_b64": ai_audio_b64
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

