import io
import os
import wave
import sounddevice as sd
from google.cloud import speech_v2

# ===== CONFIG =====
PROJECT_ID = "spring-radar-474120-c4"   # 🔹 Replace with your Google Cloud project ID
RATE = 16000                     # Audio sampling rate (Hz)
DURATION = 5                     # Recording length (seconds)

# Make sure your credentials are set:
# export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account.json"

def record_audio() -> bytes:
    """Record from microphone and return as WAV bytes."""
    print("🎙️ Speak now...")
    audio = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1, dtype="int16")
    sd.wait()
    print("✅ Recorded. Sending to Google...")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()

def transcribe_audio(audio_bytes: bytes):
    """Send audio to Google Speech-to-Text and print transcription."""
    client = speech_v2.SpeechClient()

    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model="latest_short",
    )

    request = speech_v2.RecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/_",
        config=config,
        content=audio_bytes,
    )

    response = client.recognize(request=request)
    for result in response.results:
        print("🗣️ Transcript:", result.alternatives[0].transcript)

if __name__ == "__main__":
    audio_data = record_audio()
    transcribe_audio(audio_data)
