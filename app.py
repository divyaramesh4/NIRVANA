from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from deepface import DeepFace
import google.generativeai as genai
from gtts import gTTS
import os
import time
import glob
import logging
app = Flask(__name__, static_folder='static')
logging.basicConfig(level=logging.INFO)
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    app.logger.warning("Using a placeholder API key. Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=API_KEY)
try:
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
except Exception as e:
    app.logger.error(f"Failed to initialize GenerativeModel: {e}")
    model = None 
if not os.path.exists('static'):
    os.makedirs('static')
SUICIDE_KEYWORDS = ["I want to die", "I want to end my life", "I keep harming myself", "I like the pain", "kill myself", "wish I was dead", "no reason to live"]
SUICIDE_RESPONSE = """It sounds like you're going through a very difficult time, and it's important to reach out for immediate support. Please know that you're not alone and there are people who care and want to help.

Here are some suicide helpline numbers in India you can contact right now:

- **AASRA:** +91 98204 66726
- **Vandrevala Foundation for Mental Health:** 1860-2662-345 and 1800-2333-330 (24x7)
- **iCALL:** 022-25521111 (Monday to Saturday, 8 AM to 10 PM)
- **COOJ Mental Health Foundation (Goa):** 0832-2252525
- **Connecting Trust (Pune):** +91 9922001122, +91 9922004305

Please seek professional help as soon as possible. Your well-being is important, and there are professionals who are trained to provide the support you need. We care about you."""

def generate_response(user_input, emotion, is_first):
    if not model:
        return "Chatbot model is not available. Please check API key and configuration."
    lowercased_input = user_input.lower()
    for keyword in SUICIDE_KEYWORDS:
        if keyword in lowercased_input:
            return SUICIDE_RESPONSE

    if is_first:
        prompt = f"""
        The user's initial facial expression suggests they might be feeling: **{emotion}**.

        Your job is to:
        - Greet the user with a very short, warm, and empathetic message based on this initial emotional cue.
        -Don't ask questions
        - Depending on the input you get the response the sentences should limited to 3-4 sentences or below.
        - Act like a professional therapist. Avoid responses which are too casual or informal, you can in few cases invlove humour.
        - Remember previous conversation.
        - Avoid including emojis in responses.
        - If the user asks a question, try to answer it based on.
        - If the user expresses a feeling, acknowledge it.
        - If the user is feeling anxious, suggest a quick breathing exercise.
        - If the user is feeling sad, suggest a quick positive affirmation.
        - If the user is feeling frustrated, suggest a quick grounding technique.
        - If the user is feeling happy, suggest a way to celebrate that feeling.
        - If the user is feeling angry, suggest a quick anger management technique.
        Output only the chatbot's message.
        """
    else:
        prompt = f"""
        The user's initial emotion was detected as: {emotion}.
        The user just said: "{user_input}"

        Your job is to:
        - Consider the user's input
        -Don't ask questions
        - Avoid including emojis in responses.
        - Act like a professional therapist. Avoid responses which are too casual or informal, you can in few cases invlove humour.
        - Respond concisely while keeping the tone warm, friendly and supportive.
        - Avoid long responses. Keep it short and engaging (1-3 sentences).
        - If the user's input is a simple greeting, respond in kind.
        - If the user asks a question, try to answer it.
        - If the user expresses a feeling, acknowledge it.
        Output only the chatbot's message.
        """

    try:
        chat = model.start_chat() # Consider if chat history should be maintained for longer conversations
        response = chat.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        app.logger.error(f"Error generating response from AI model: {str(e)}")
        return f"Oops! I had a little trouble thinking. 😢 ({str(e)})"


def text_to_speech(response_text):
    if not response_text:
        return None
    try:
        tts = gTTS(response_text)
        # Using a timestamp for unique filenames to avoid caching issues
        timestamp = int(time.time() * 1000)
        filename = f"response_{timestamp}.mp3"
        audio_path = os.path.join("static", filename)
        tts.save(audio_path)
        app.logger.info(f"Generated audio file: {audio_path}")
        return filename
    except Exception as e:
        app.logger.error(f"Error generating text-to-speech: {str(e)}")
        return None


def cleanup_old_audio_files():
    try:
        files = glob.glob('static/response_*.mp3')
        now = time.time()
        for file_path in files:
            if os.path.isfile(file_path):
                # Delete files older than 1 hour (3600 seconds)
                if now - os.path.getmtime(file_path) > 3600:
                    os.remove(file_path)
                    app.logger.info(f"Cleaned up old audio file: {file_path}")
    except Exception as e:
        app.logger.error(f"Error during audio file cleanup: {str(e)}")
@app.route('/')
def landing_page():
    return render_template('landingindex.html')

@app.route('/navigated')
def navigated_page():
 return render_template('navigated.html')

@app.route('/home')
def homepage():
 return render_template('homepageindex.html')



@app.route('/detect-emotion', methods=['POST'])
def detect_emotion_route():
    cleanup_old_audio_files()
    image_data_uri = request.json.get('image')
    if not image_data_uri:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        image_data = image_data_uri.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
             return jsonify({'error': 'Could not decode image'}), 400

        analysis_results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)

        if isinstance(analysis_results, list) and len(analysis_results) > 0:
            emotion = analysis_results[0]['dominant_emotion']
        else: 
            emotion = "unknown"
        app.logger.info(f"Detected emotion: {emotion}")

    except Exception as e:
        app.logger.error(f"Error during emotion detection: {str(e)}")
        emotion = "error" 
    response_text = generate_response("...", emotion, is_first=True)
    audio_filename = text_to_speech(response_text)
    audio_url = f"/static/{audio_filename}" if audio_filename else None

    return jsonify({'response': response_text, 'emotion': emotion, 'audio_url': audio_url})


@app.route('/chat', methods=['POST'])
def chat_route():
    cleanup_old_audio_files() 
    data = request.json
    user_input = data.get("message", "")
    emotion = data.get("emotion", "neutral") 

    response_text = generate_response(user_input, emotion, is_first=False)
    audio_filename = text_to_speech(response_text)
    audio_url = f"/static/{audio_filename}" if audio_filename else None

    return jsonify({'response': response_text, 'audio_url': audio_url}) 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)