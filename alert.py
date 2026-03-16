from twilio.rest import Client
from config.alerts import (
    TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
    TWILIO_FROM, OWNER_WHATSAPP,
    CLOUDINARY_CLOUD_NAME, CLOUDINARY_UPLOAD_PRESET,
)
import os
import requests


def _upload_to_cloudinary(image_path: str) -> str:
    """
    Upload image via Cloudinary unsigned upload.
    Returns public URL or empty string on failure.
    Requires a free Cloudinary account + unsigned upload preset.
    """
    if not CLOUDINARY_CLOUD_NAME or not CLOUDINARY_UPLOAD_PRESET:
        print("[WARN] Cloudinary not configured — sending text-only alert")
        return ""
    try:
        url = f"https://api.cloudinary.com/v1_1/{CLOUDINARY_CLOUD_NAME}/image/upload"
        with open(image_path, "rb") as f:
            resp = requests.post(
                url,
                data={"upload_preset": CLOUDINARY_UPLOAD_PRESET},
                files={"file": f},
                timeout=20,
            )
        if resp.status_code == 200:
            img_url = resp.json()["secure_url"]
            print(f"[ALERT] Image uploaded: {img_url}")
            return img_url
        else:
            print(f"[WARN] Cloudinary upload failed ({resp.status_code}): {resp.text[:200]}")
            return ""
    except Exception as e:
        print(f"[WARN] Cloudinary upload error: {e}")
        return ""


def send_whatsapp_alert(snap_path: str, score: int, tid: int):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        body = (
            f"🚨 SHOPLIFTING ALERT!\n"
            f"Suspicious activity detected.\n"
            f"Score: {score}/100\n"
            f"Time: {os.path.basename(snap_path)}"
        )

        msg_kwargs = {"from_": TWILIO_FROM, "to": OWNER_WHATSAPP, "body": body}

        if os.path.exists(snap_path):
            img_url = _upload_to_cloudinary(snap_path)
            if img_url:
                msg_kwargs["media_url"] = [img_url]

        message = client.messages.create(**msg_kwargs)
        print(f"[ALERT] WhatsApp sent — SID: {message.sid}")

    except Exception as e:
        print(f"[ERROR] WhatsApp alert failed: {e}")


def start_ngrok(snapshot_dir: str):
    pass  # no longer needed
