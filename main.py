from fastapi import FastAPI, UploadFile, Form
from typing import Optional
from fastapi.responses import FileResponse
import os
import json
import re
import uuid
import PyPDF2
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import GenerativeModel
import requests
from pydub import AudioSegment

load_dotenv()

# === CONFIG ===
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_URL = os.getenv("ELEVENLABS_URL")
VOICE_ONE_ID = os.getenv("VOICE_ONE_ID")  # Nadya
VOICE_TWO_ID = os.getenv("VOICE_TWO_ID")  # Alif

# === INIT ===
genai.configure(api_key=GENAI_API_KEY)
app = FastAPI()

# === FUNCTIONS ===
def read_pdf_to_text(file_path):
    text = ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def clean_text(text):
    import re
    text = text.replace("*", "")
    text = re.sub(r'"', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def synthesize_speech(text, voice_id, output_path):
    url = f"{ELEVENLABS_URL}/{voice_id}/stream"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": clean_text(text),
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.3
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Error generating speech: {response.text}")

# === MAIN ENDPOINT ===
@app.post("/generate-podcast")
async def generate_podcast(
    file: Optional[UploadFile] = None,
    topic_text: Optional[str] = Form(None)
):
    if file:
        input_filename = f"temp_{uuid.uuid4().hex}.pdf"
        with open(input_filename, "wb") as f:
            f.write(await file.read())
        input_text = read_pdf_to_text(input_filename)
        os.remove(input_filename)
    elif topic_text:
        input_text = topic_text
    else:
        return {"error": "Kirimkan file PDF atau masukkan teks topik"}

    prompt = f"""
    Kamu adalah seorang penulis naskah podcast profesional yang menulis dalam Bahasa Indonesia. 
    Tugasmu adalah membuat dialog singkat bergaya podcast antara dua pembicara: Nadya dan Alif.
    Gunakan gaya santai sehari-hari. Balasanmu harus berupa JSON array dengan field 'speaker' dan 'line'.
    Berdasarkan informasi berikut:

    {input_text}
    """

    model = GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.9,
            "top_p": 0.95,
            "max_output_tokens": 4096
        }
    )

    try:
        match = re.search(r"\[\s*{.*}\s*\]", response.text, re.DOTALL)
        if match:
            json_str = match.group(0)
            dialogue = json.loads(json_str)
        else:
            dialogue = []
    except Exception as e:
        return {"error": f"Parsing JSON gagal: {str(e)}"}

    if not dialogue:
        return {"error": "Tidak ada dialog yang dihasilkan"}

    # === Buat audio ===
    os.makedirs("podcast_audio", exist_ok=True)
    os.makedirs("final_output", exist_ok=True)
    audio_files = []

    for idx, turn in enumerate(dialogue):
        speaker = turn["speaker"].strip().lower()
        line = turn["line"]
        voice_id = VOICE_ONE_ID if speaker == "nadya" else VOICE_TWO_ID
        output_file = f"podcast_audio/turn_{idx}_{speaker}.mp3"
        synthesize_speech(line, voice_id, output_file)
        audio_files.append(output_file)

    # Gabung audio jadi satu
    podcast = AudioSegment.empty()
    for file in audio_files:
        segment = AudioSegment.from_file(file)
        podcast += segment + AudioSegment.silent(duration=500)

    output_final_path = f"final_output/podcast_final_{uuid.uuid4().hex}.mp3"
    podcast.export(output_final_path, format="mp3")

    return FileResponse(output_final_path, media_type="audio/mpeg", filename=os.path.basename(output_final_path))