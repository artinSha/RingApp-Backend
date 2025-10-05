import io, os
from google.cloud import speech_v2
from google.oauth2 import service_account
from google.auth.transport.requests import Request

PROJECT_ID = os.getenv("PROJECT_ID", "spring-radar-474120-c4")
KEY_PATH = "/Users/andywoochanjung/Desktop/RingApp-Backend/config/spring-radar-474120-c4-70f2fa862484.json"
CLOUD_SCOPE = ["https://www.googleapis.com/auth/cloud-platform"]

def transcribe_wav(path: str):
    # 1) Load + scope creds, set quota project
    base = service_account.Credentials.from_service_account_file(KEY_PATH)
    scoped = base.with_scopes(CLOUD_SCOPE).with_quota_project(PROJECT_ID)

    # (optional) sanity: refresh and show token slice
    scoped.refresh(Request())
    print("🔐 SA:", scoped.service_account_email, "| token starts:", scoped.token[:12], "...")

    # 2) Use these creds in the Speech client
    client = speech_v2.SpeechClient(credentials=scoped)

    with open(path, "rb") as f:
        audio_bytes = f.read()

    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model="latest_long",
    )

    req = speech_v2.RecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/_",
        config=config,
        content=audio_bytes,
    )
    resp = client.recognize(request=req)

    if not resp.results:
        print("❌ No transcription results.")
        return

    for r in resp.results:
        if r.alternatives:
            print("🗣️", r.alternatives[0].transcript)

if __name__ == "__main__":
    transcribe_wav("maybe-next-time.wav")
