from flask import Flask, request, jsonify
from comment import get_comment, clip_video,get_video_image_sequence
from utils import load_image
from werkzeug.utils import secure_filename
import os
import cv2
import json

app = Flask(__name__)


@app.route("/comment", methods=['POST'])
def explain_video(): 
    if request.method == 'POST':
        prompt = request.form['prompt']
        nucleus_sampling = request.form['nucleus_sampling']
        temperature = request.form['temperature']
        top_percent = request.form['top_percent']
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                if not os.path.exists("./tmp"):
                    os.makedirs("./tmp")
                file.save(os.path.join('./tmp/', filename))
        image = load_image(f'./tmp/{filename}')
        response = {}
        response['comment'] = get_comment(image,
                                           prompt=prompt,
                                           nucleus_sampling=nucleus_sampling,
                                           temperature=temperature,
                                           top_percent=top_percent)
        return jsonify(response)

@app.route("/describe", methods=['POST'])
def describe_single_image(): 
    if request.method == 'POST':
        nucleus_sampling = False
        temperature = 0.2
        top_percent = 0.9
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                if not os.path.exists("./tmp"):
                    os.makedirs("./tmp")
                file.save(os.path.join('./tmp/', filename))
        image = load_image(f'./tmp/{filename}')
        response = {}
        response['description'] = get_comment(image,
                                           nucleus_sampling=nucleus_sampling,
                                           temperature=temperature,
                                           top_percent=top_percent)
        return jsonify(response)

@app.route("/timing", methods=['POST'])
def automatic_timing():
    if request.method == 'POST':
        video = request.form['video']
        scenes = json.loads(request.form['scenes']) #(start_time, end_time) time in 100's of second
        timing = []
        nucleus_sampling = False
        temperature = 0.2
        top_percent = 0.9
        for i,scene in enumerate(scenes):
            try:
                clip = clip_video(f"../search-api/videos/{video}",scene,f'{i}')
                image_file = get_video_image_sequence(f'./tmp/{clip}',f'{i}')
                image = load_image(f'./tmp/{image_file}')
                comment = get_comment(image,
                            nucleus_sampling=nucleus_sampling,
                            temperature=temperature,
                            top_percent=top_percent)
                print(f'{i} -- {comment}')
                file1 = open(f"./tmp/{video}.txt", "a")  # append mode
                file1.write(f'{i} -- {comment} \n')
                file1.close()
                timing.append(
                    dict(
                        start=scene[0],
                        end=scene[1],
                        text=comment
                    )
                )
            except:
                print(f"ERROR start time {scene[0]} end time {scene[1]}")
        response = {}
        response['timing'] = timing
        return jsonify(response)
    
if __name__ == "__main__":
    app.run(debug=True, use_debugger=False, use_reloader=False,port=5002)