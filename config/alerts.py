import os
from dotenv import load_dotenv

load_dotenv()

TWILIO_ACCOUNT_SID       = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN        = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM              = os.getenv("TWILIO_FROM", "")
OWNER_WHATSAPP           = os.getenv("OWNER_WHATSAPP", "")
CLOUDINARY_CLOUD_NAME    = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_UPLOAD_PRESET = os.getenv("CLOUDINARY_UPLOAD_PRESET", "")
