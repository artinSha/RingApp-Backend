from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
from bson.objectid import ObjectId
from flask_cors import CORS
import requests
import base64

# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

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
    scenario = data.get("scenario", "General")

    if not user_id:
        return jsonify({"error": "user_id required"}), 400

    # Verify user exists
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        return jsonify({"error": "invalid user_id"}), 400

    # Create conversation document
    conv_doc = {
        "user_id": user_id,
        "scenario": scenario,
        "conversation": [],  # store AI+user turns
        "timestamp": datetime.now(timezone.utc),  # Changed here
        "grammar_feedback": None
    }
    conv_res = conversations_collection.insert_one(conv_doc)
    conv_id = str(conv_res.inserted_id)

    # -----------------------
    # Always generate first AI line
    # -----------------------
    ai_text = "Hello! This is a placeholder AI line for your scenario."
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

# -------------------------------
# Helper: Transcription of user audio using Google Cloud STT
# -------------------------------
def transcribe_audio_stt(audio_file):
    """
    Replace with Gemini/OpenAI transcription later.
    Currently just returns dummy text.
    """
    return "This is a placeholder transcription of user audio."


# -------------------------------
# Helper: Generates AI response based on string of conversation history thus far
# -------------------------------
def generate_ai_text(conversation_context):
    """
    Replace with Gemini API call later.
    Currently just returns dummy AI text.
    """
    return "Great! You should look for a safe spot immediately."


# -------------------------------
# Helper: ElevenLabs TTS
# -------------------------------
def elevenlabs_tts(text):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {"text": text}
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"TTS failed: {response.text}")
    return response.content  # returns bytes

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

    # Generate TTS via ElevenLabs
    ai_audio_bytes = elevenlabs_tts(ai_text)
    ai_audio_b64 = base64.b64encode(ai_audio_bytes).decode("utf-8")  # frontend can use data URI

    # Return response to frontend
    return jsonify({
        "user_text": user_text,
        "ai_text": ai_text,
        "ai_audio_b64": ai_audio_b64
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
