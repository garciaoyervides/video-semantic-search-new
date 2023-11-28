from transcribe import get_transcription
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/transcribe")
def transcribe_and_segment():
    video_file_name = request.args.get('video')
    video_language = request.args.get('lang')
    if video_file_name and video_language:
        response = {}
        transcript, translated_transcript = get_transcription(video_file_name,video_language)
        response['transcript'] = transcript
        response['translated_transcript'] = translated_transcript
        return jsonify(response)
    else:
        return jsonify("Please send a video file name")

if __name__ == "__main__":
    app.run(debug=True, use_debugger=False, use_reloader=False,port=5001)