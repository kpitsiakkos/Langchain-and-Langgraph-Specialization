import os
import requests
import whisper  # type: ignore[import-untyped]
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Environment variables ─────────────────────────────────────────────────────
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
ZAPIER_WEBHOOK_URL = os.getenv("ZAPIER_WEBHOOK_URL")

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY is not set.")
if not ZAPIER_WEBHOOK_URL:
    raise EnvironmentError("ZAPIER_WEBHOOK_URL is not set.")


# ── Transcription ─────────────────────────────────────────────────────────────
def transcribe(file_path: str, model_size: str = "base") -> str:
    """Transcribe an audio file using OpenAI Whisper."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    model  = whisper.load_model(model_size)
    result = model.transcribe(file_path)
    return result["text"].strip()


# ── Summarisation ─────────────────────────────────────────────────────────────
def summarise(transcript: str) -> str:
    """Generate a concise summary of a call transcript using GPT."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a customer care assistant. Summarise the following "
                    "call transcript in clear, concise bullet points covering: "
                    "key topics discussed, any issues raised, and action items "
                    "or resolutions agreed upon."
                ),
            },
            {"role": "user", "content": transcript},
        ],
    )
    return response.choices[0].message.content


# ── Zapier webhook ────────────────────────────────────────────────────────────
def trigger_zapier(recipient_email: str, subject: str, body: str) -> None:
    """POST summary data to a Zapier webhook → Gmail action."""
    payload  = {"to": recipient_email, "subject": subject, "body": body}
    response = requests.post(ZAPIER_WEBHOOK_URL, json=payload, timeout=10)
    response.raise_for_status()


# ── Main entry point ──────────────────────────────────────────────────────────
def email_summary(file_path: str, recipient_email: str, model_size: str = "base") -> None:
    """Transcribe → summarise → send via Zapier."""
    transcript = transcribe(file_path, model_size)

    if not transcript:
        raise ValueError(f"Transcription returned empty text for: {file_path}")

    summary = summarise(transcript)

    trigger_zapier(
        recipient_email=recipient_email,
        subject="Customer Care Call Summary",
        body=summary,
    )