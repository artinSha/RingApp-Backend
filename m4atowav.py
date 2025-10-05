import os
import io
import tempfile
import ffmpeg
from google.cloud import speech_v2
from google.oauth2 import service_account
from google.auth.transport.requests import Request

# === CONFIG ===
PROJECT_ID = "spring-radar-474120-c4"
KEY_PATH = "/Users/andywoochanjung/Desktop/RingApp-Backend/config/spring-radar-474120-c4-70f2fa862484.json"
CLOUD_SCOPE = ["https://www.googleapis.com/auth/cloud-platform"]


def convert_m4a_to_wav(m4a_path: str) -> str:
    """Convert .m4a to temporary .wav file using ffmpeg."""
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()

    try:
        (
            ffmpeg
            .input(m4a_path)
            .output(tmp_wav.name, format="wav", acodec="pcm_s16le", ac=1, ar="16000")
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"🎧 Converted {m4a_path} → {tmp_wav.name}")
        return tmp_wav.name
    except ffmpeg.Error as e:
        print("❌ ffmpeg conversion failed:", e)
        raise


def transcribe_audio(wav_path: str):
    """Transcribe a WAV file using Google Cloud Speech-to-Text v2."""
    # load + refresh credentials
    base = service_account.Credentials.from_service_account_file(KEY_PATH)
    scoped = base.with_scopes(CLOUD_SCOPE).with_quota_project(PROJECT_ID)
    scoped.refresh(Request())
    print("🔐 Authenticated as:", scoped.service_account_email)

    # set up client
    client = speech_v2.SpeechClient(credentials=scoped)

    with io.open(wav_path, "rb") as f:
        audio_bytes = f.read()

    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model="latest_long",  # use long for >15s audio
    )

    request = speech_v2.RecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/_",
        config=config,
        content=audio_bytes,
    )

    response = client.recognize(request=request)

    print("\n=== TRANSCRIPTION ===")
    if not response.results:
        print("❌ No results found.")
    for r in response.results:
        if r.alternatives:
            print("🗣️", r.alternatives[0].transcript)
    print("=====================\n")


if __name__ == "__main__":
    m4a_path = "Recording (3).m4a"  # replace with your incoming file path
    wav_path = convert_m4a_to_wav(m4a_path)
    transcribe_audio(wav_path)
    os.remove(wav_path)  # cleanup temporary wav
