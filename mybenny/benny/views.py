import os
import io
import time
import base64
from dotenv import load_dotenv
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse,StreamingHttpResponse
from django.core.files.uploadedfile import InMemoryUploadedFile
import requests
import uuid
from pydub import AudioSegment
from openai import OpenAI
from .util import *
from .models import LocationUser, ChatLog, ChatSession



load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")

memory = ConversationBufferMemory(return_messages=True)
last_interaction_time = time.time()


def homepage(request):
    if "chat_session_id" not in request.session:
        return redirect("login")
    return render(request, "voices.html")


@csrf_exempt
@csrf_exempt
def transcribe(request):
    global last_interaction_time, memory

    if request.method != 'POST':
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    now = time.time()
    if now - last_interaction_time > 60:
        memory.clear()
    last_interaction_time = now

    audio_file = request.FILES.get("audio")
    if not audio_file:
        return JsonResponse({"error": "No audio file provided."}, status=400)

    # Convert webm to wav
    audio_bytes = audio_file.read()
    webm_io = io.BytesIO(audio_bytes)
    audio = AudioSegment.from_file(webm_io, format="webm")
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav", codec="pcm_s16le", parameters=["-ar", "16000", "-ac", "1"])
    wav_io.seek(0)
    wav_io.name = "audio.wav"

    # Transcribe using Whisper
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=wav_io,
        response_format="text"
    )
    user_text = transcription.strip()

    # Get LLM response
    response_text = generate_response(user_text, memory)

    # Save to DB
    session_id = request.session.get("chat_session_id")
    if session_id:
        try:
            session = ChatSession.objects.get(session_id=session_id)
            ChatLog.objects.create(
                session=session,
                question=user_text,
                answer=response_text
            )
        except ChatSession.DoesNotExist:
            print("⚠️ ChatSession not found. Log not saved.")
    else:
        print("⚠️ No chat_session_id in session. Log not saved.")

    # Prepare ElevenLabs TTS request
    headers = {
        "xi-api-key": elevenlabs_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": response_text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    # Make streamed TTS request
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    tts_response = requests.post(tts_url, headers=headers, json=payload, stream=True)

    # Return streamed audio as response
    def stream_audio():
        for chunk in tts_response.iter_content(chunk_size=4096):
            if chunk:
                yield chunk

    response = StreamingHttpResponse(
        streaming_content=stream_audio(),
        content_type='audio/mpeg'
    )
    response["X-Transcription"] = user_text
    response["X-Response-Text"] = response_text
    return response



def login_view(request):
    locations = LocationUser.objects.all()

    if request.method == "POST":
        new_location = request.POST.get("new_location", "").strip()
        selected_location = request.POST.get("existing_location", "").strip()

        location_name = new_location or selected_location

        if not location_name:
            return render(request, "login.html", {
                "error": "Please enter or select a location",
                "locations": locations
            })

        # Create or get the location
        user, _ = LocationUser.objects.get_or_create(name=location_name)

        # ✅ Create a new ChatSession for this login
        session_uuid = str(uuid.uuid4())
        chat_session = ChatSession.objects.create(location=user, session_id=session_uuid)

        # ✅ Store both in Django session
        request.session["user_id"] = user.id
        request.session["chat_session_id"] = session_uuid

        return redirect("homepage")

    return render(request, "login.html", {"locations": locations})



def logout_view(request):
    request.session.flush()  # 🔐 This clears all session data
    return redirect("login")  

# def transcribe(request):
#     global last_interaction_time, memory

#     if request.method == 'POST':
#         now = time.time()
#         if now - last_interaction_time > 60:
#             memory.clear()
#         last_interaction_time = now

#         audio_file: InMemoryUploadedFile = request.FILES.get("audio")
#         if not audio_file:
#             return JsonResponse({"error": "No audio file provided."}, status=400)

#         audio_bytes = audio_file.read()
#         webm_io = io.BytesIO(audio_bytes)
#         audio = AudioSegment.from_file(webm_io, format="webm")
#         wav_io = io.BytesIO()
#         audio.export(wav_io, format="wav", codec="pcm_s16le", parameters=["-ar", "16000", "-ac", "1"])
#         wav_io.seek(0)
#         wav_io.name = "audio.wav"

#         transcription = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=wav_io,
#             response_format="text"
#         )
#         user_text = transcription.strip()

#         response_text = generate_response(user_text, memory)

#         tts_response = client.audio.speech.create(
#             model="tts-1",
#             voice="ash",
#             input=response_text
#         )

#         audio_base64 = base64.b64encode(tts_response.content).decode("utf-8")

#         return JsonResponse({
#             "transcription": user_text,
#             "response_text": response_text,
#             "audio_base64": audio_base64
#         })

#     return JsonResponse({"error": "Only POST allowed"}, status=405)
