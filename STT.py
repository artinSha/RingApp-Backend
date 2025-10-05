import io, os
from google.cloud import speech_v2
from google.oauth2 import service_account
from google.auth.transport.requests import Request

PROJECT_ID = os.getenv("PROJECT_ID", "spring-radar-474120-c4")

# # Get the directory where this script is located, then go to the key file
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# KEY_PATH = os.path.join(BASE_DIR, "config", "spring-radar-474120-c4-70f2fa862484.json")

CLOUD_SCOPE = ["https://www.googleapis.com/auth/cloud-platform"]

def _load_credentials():
    """
    Load credentials from the Railway environment variable GCP_KEY_JSON.
    Falls back to GOOGLE_APPLICATION_CREDENTIALS if present.
    """
    key_json = os.getenv("GCP_KEY_JSON")
    if key_json:
        try:
            info = json.loads(key_json)
            creds = service_account.Credentials.from_service_account_info(info)
        except json.JSONDecodeError:
            raise RuntimeError("❌ GCP_KEY_JSON is not valid JSON.")
    elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        )
    else:
        raise RuntimeError("❌ No Google credentials found. Set GCP_KEY_JSON in Railway.")

    # Add scopes + project quota
    scoped = creds.with_scopes(CLOUD_SCOPE).with_quota_project(PROJECT_ID)
    return scoped

# def transcribe_wav(path: str):
#     # 1) Load + scope creds, set quota project
#     base = service_account.Credentials.from_service_account_file(KEY_PATH)
#     scoped = base.with_scopes(CLOUD_SCOPE).with_quota_project(PROJECT_ID)

#     # (optional) sanity: refresh and show token slice
#     scoped.refresh(Request())
#     print("🔐 SA:", scoped.service_account_email, "| token starts:", scoped.token[:12], "...")

#     # 2) Use these creds in the Speech client
#     client = speech_v2.SpeechClient(credentials=scoped)

#     with open(path, "rb") as f:
#         audio_bytes = f.read()

#     config = speech_v2.RecognitionConfig(
#         auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
#         language_codes=["en-US"],
#         model="latest_long",
#     )

#     req = speech_v2.RecognizeRequest(
#         recognizer=f"projects/{PROJECT_ID}/locations/global/recognizers/_",
#         config=config,
#         content=audio_bytes,
#     )
#     resp = client.recognize(request=req)

#     if not resp.results:
#         print("❌ No transcription results.")
#         return

#     for r in resp.results:
#         if r.alternatives:
#             print("🗣️", r.alternatives[0].transcript)

def transcribe_wav(path: str) -> str:
    """
    Transcribes an audio file (WAV, M4A, etc.) using Google Cloud Speech-to-Text v2.
    Works on Railway using credentials from the GCP_KEY_JSON environment variable.
    """
    # 1) Load creds and refresh token
    scoped = _load_credentials()
    scoped.refresh(Request())
    print("🔐 Using SA:", scoped.service_account_email)

    # 2) Initialize Speech client
    client = speech_v2.SpeechClient(credentials=scoped)

    # 3) Read audio file bytes
    with open(path, "rb") as f:
        audio_bytes = f.read()

    # 4) Configure request
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

    # 5) Send to API and parse results
    resp = client.recognize(request=req)
    if not resp.results:
        print("❌ No transcription results.")
        return ""

    text_parts = []
    for r in resp.results:
        if r.alternatives:
            text_parts.append(r.alternatives[0].transcript.strip())

    transcript = " ".join(text_parts)
    print("🗣️ Transcript:", transcript)
    return transcript
