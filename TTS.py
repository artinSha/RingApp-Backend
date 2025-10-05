import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID")


def elevenlabs_tts_get_bytes(text, voice_id=None) -> bytes:
    """Call ElevenLabs and return raw mp3 bytes. Raises on failure."""
    voice = voice_id or ELEVEN_VOICE_ID
    if not ELEVENLABS_API_KEY or not voice:
        raise RuntimeError("ELEVENLABS_API_KEY or ELEVEN_VOICE_ID not set in env")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "voice_settings": {"stability": 0.6, "similarity_boost": 0.75}
    }

    resp = requests.post(url, headers=headers, json=payload, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs TTS failed ({resp.status_code}): {resp.text}")

    return resp.content