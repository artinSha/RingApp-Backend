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
import re

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
from scenarios import (
    SCENARIOS,
    find_scenario_key_by_title,
    get_model_for_scenario,
    gemini_opening_for_scenario,
    get_evaluator_model
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

    scenario_key = find_scenario_key_by_title(scenario_title)

    # Create conversation document
    conv_doc = {
        "user_id": user_id,
        "scenario": scenario_key,
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

    ai_text = gemini_opening_for_scenario(scenario_key)

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
        # Generate TTS via ElevenLabs and return base64
    try:
        ai_audio_bytes = elevenlabs_tts_get_bytes(ai_text)
        ai_audio_b64 = base64.b64encode(ai_audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"TTS Error: {str(e)}")  # Log the actual error
        return jsonify({"error": f"TTS generation failed: {str(e)}"}), 500

    payload = {
        "conversation_id": conv_id,
        "initial_ai_text": ai_text,
        "initial_ai_audio_b64": ai_audio_b64
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

def _extract_user_utterances(conversation_doc, max_chars: int = 6000) -> str:
    """
    Return a single string with only the user's utterances, newest last.
    Soft-limit total chars to keep prompts reasonable.
    """
    buf = []
    total = 0
    for t in conversation_doc.get("conversation", []):
        ut = (t.get("user_text") or "").strip()
        if ut:
            if total + len(ut) > max_chars:
                # Simple truncation from the start if too long
                # Keep the most recent parts by popping from the beginning
                while buf and (total + len(ut) > max_chars):
                    removed = buf.pop(0)
                    total -= len(removed)
            buf.append(ut)
            total += len(ut)
    return "\n".join(f"- {u}" for u in buf) if buf else "(no user speech captured)"


def _build_feedback_prompt(conv_id: str, scenario_key: str, user_utterances: str) -> str:
    """
    ESL tutor-style evaluation prompt.
    Returns strictly-JSON guidance: CEFR, TOEFL estimate, strengths, issues,
    concise corrections, and short practice tips.
    """
    # Retrieve conversation from DB
    convo = conversations_collection.find_one({"_id": ObjectId(conv_id)})
    if not convo:
        raise ValueError("Conversation not found in database.")
    
    s = SCENARIOS.get(scenario_key, {})
    title = s.get("title", scenario_key)
    setting = s.get("setting", "")
    stakes = s.get("stakes", "")
    role = s.get("role", "")

    return (
        "You are an experienced ESL evaluator. Analyze ONLY the learner’s utterances.\n"
        "STRICTLY with valid JSON, using only the keys below — no prose, no markdown, and no explanations.\n\n"
        "Your JSON MUST follow this schema exactly:\n"
        "{\n"
        '  "success_percentage": int (0-100),\n'
        '  "grammar_feedback": [\n'
        '    {"before": "<learner_sentence_with_error>", "after": "<corrected_sentence>"},\n'
        '    ...\n'
        "  ] OR null if there are no grammatical issues,\n"
        '  "grammar_issues": int (number of actual grammar mistakes found, 0 if none),\n'
        '   "turns": int (number of user utterances analyzed),\n'
        "}\n\n"
        "Rules and guidance:\n"
        "- Only identify grammar or phrasing issues that are clearly incorrect — do not overcorrect.\n"
        "- DO NOT invent errors. Only include grammar or phrasing errors in the following list: (tense, agreement, prepositions, articles, word order, unnatural phrasing).\n"
        "- If all utterances are grammatically correct and natural, set grammar_feedback to null and grammar_issues to 0.\n"
        "- Use simple, natural English corrections.\n"
        "- Base your judgment solely on learner utterances — ignore AI lines.\n"
        "- Return the JSON directly with no text outside the object.\n\n"
        f"Scenario Context:\n"
        f"- Title: {title}\n"
        f"- Setting: {setting}\n"
        f"- Stakes: {stakes}\n"
        f"- Roles: {role}\n\n"
        f"Learner’s utterances:\n{user_utterances}\n\n"
    )


def _clean_text(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s or "").strip().lower()

@app.route("/process_practice", methods=["POST"])
def process_practice():
    """
    Compare what the user said (from m4a) with the provided correct sentence.
    Takes (multipart/form-data):
      - audio (m4a or wav file)
      - correct_text (string)
    Returns:
      - matched (boolean)
      - spoken_text (transcribed user speech)
      - correct_text (the expected phrase)
    """
    audio_file = request.files.get("audio")
    correct_text = request.form.get("correction_text")  

    # Validate inputs first
    if not audio_file or not correct_text:
        return jsonify({"error": "audio and correct_text are required"}), 400

    # Transcribe (m4a will be converted inside transcribe_audio_stt)
    try:
        spoken_text = transcribe_audio_stt(audio_file) or ""
    except Exception as e:
        return jsonify({"error": f"Transcription failed: {e}"}), 500

    # Case-insensitive trimmed comparison
    matched = _clean_text(spoken_text) == _clean_text(correct_text)

    return jsonify({
        "matched": matched,
    }), 200


# ------------------------- Gemini AI setup
# -------------------------
configure_genai()  # uses GEMINI_API or GEMINI_API_KEY
    
def generate_ai_text(conversation_context: str, scenario_key: str = "General") -> str:
    """
    Generates the next AI reply from Gemini using a single string prompt.
    This keeps compatibility with your current /process_audio call.
    """
    try:
        model = get_model_for_scenario(scenario_key)
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

    #Transcribe audio using speech to text
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
    ai_text = generate_ai_text(context_text, scenario_key=conversation.get("scenario", "General"))

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

from datetime import datetime


#End call endpoint to send conversation
@app.route("/end_call", methods=["POST"])
def end_call():
    conv_id = request.form.get("conv_id") or (request.json or {}).get("conv_id")
    if not conv_id:
        return jsonify({"error": "conv_id is required"}), 400

    # 1) Load conversation
    convo = conversations_collection.find_one({"_id": ObjectId(conv_id)})
    if not convo:
        return jsonify({"error": "Invalid conversation ID"}), 400

    scenario_key = convo.get("scenario", "General")

    # 2) Build feedback prompt (scenario + only user lines)
    user_utterances = _extract_user_utterances(convo)
    prompt = _build_feedback_prompt(conv_id, scenario_key, user_utterances)

    # 3) Ask Gemini for JSON feedback using the SCENARIO model
    try:
        model = get_evaluator_model()
        resp = model.generate_content(prompt)
        feedback_text = (getattr(resp, "text", "") or "").strip()
    except Exception as e:
        # If Gemini fails, store a friendly error
        feedback_text = json.dumps({
            "error": "feedback_generation_failed",
            "detail": str(e)[:300]
        })

    # 4) Try to parse JSON to ensure it’s valid; if not, wrap it
    try:
        parsed = json.loads(feedback_text)
        grammar_feedback = parsed
    except Exception:
        # Model didn’t return pure JSON; store raw text for debugging
        grammar_feedback = {"raw": feedback_text}

    # OPTIONAL: Save feedback back into the conversation
    conversations_collection.update_one(
        {"_id": ObjectId(conv_id)},
        {"$set": {"grammar_feedback": grammar_feedback}}
    )

    # Only keep the conversation array, not metadata
    conversation_array = convo.get("conversation", [])

    response = {
        "conversation_id": conv_id,
        "grammar_feedback": grammar_feedback,
        "conversation": conversation_array
    }

    return jsonify({
        "scenario": "test",
        "userTranscript": [i["user_text"] for i in conversation_array],
        "aiTranscript": [i["ai_text"] for i in conversation_array],
        "grammarErrors": [{"error": i["before"], "correction": i["after"]} for i in grammar_feedback["grammar_feedback"]],
        "score": grammar_feedback["success_percentage"],
        "encouragement": "filler" 
    }), 200



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

