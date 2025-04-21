
from flask import Flask, jsonify, Response, stream_with_context, render_template

from .util import *

app = Flask(__name__)

whisper_model, kokoro_pipeline = initialize_models()

@app.route('/')
def home():
    return render_template('index.html')


@app.route("/stream/")
def stream():
    return Response(stream_with_context(continuous_interaction(whisper_model, kokoro_pipeline)), content_type="text/event-stream")

@app.route("/question/")
def chat():
    return jsonify({"message": "Started listening..."})


@app.route("/try/")
def audio():
    return render_template('data.html')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)