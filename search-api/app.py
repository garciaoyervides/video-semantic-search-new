import base64
from flask import Flask, request, jsonify
from search import search_scene, clip_video, h_seconds_to_timestamp
from werkzeug.utils import secure_filename
import os
from upload import process_video_to_db
import threading
from databasechroma import get_logs,get_latest_log,get_video_list
import json

app = Flask(__name__,root_path='../')

@app.route("/search", methods=['POST'])
def search_video():
    if request.method == 'POST':
        k = int(request.form['k'])
        #index = request.form['index']
        index = json.loads(request.form['index'])
        if "text" in request.form:
            text = request.form['text']
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                if not os.path.exists("./tmp"):
                    os.makedirs("./tmp")
                file.save(os.path.join('./tmp/', filename))
        try:
            if "text" in request.form:
                results = search_scene(text, k=k, index=index, input_type="Text")
            if 'file' in request.files:
                results = search_scene(filename, k=k, index=index, input_type="Image")
            #the result is this ([((video_id,video_name,scene_number))],[distance])
        except Exception as error:
            print(error)
            results = None
        response = []
        if results:
            for i, id in enumerate(results['ids'][0]):
                video_name=results['metadatas'][0][i]['video name']
                distance = results['distances'][0][i]
                time_range = (results['metadatas'][0][i]['start'],results['metadatas'][0][i]['end'])
                transcript=results['metadatas'][0][i]['transcript']
                description=results['metadatas'][0][i]['description']
                scene=results['metadatas'][0][i]['scene']
                try:
                    rrf=results['rrf'][0][i]
                except:
                    rrf = 'None'
                try:
                    video = clip_video(video_name,time_range,f'segment_{str(i).zfill(3)}')
                    with open(video, "rb") as video_file:
                        encoded_string = base64.b64encode(video_file.read())
                    encoded_string = encoded_string.decode()
                except Exception as error:
                    print(error)
                    encoded_string = ""
                response.append({
                'distance': distance,
                'video': encoded_string,
                'identifier': f'Video: {video_name} Scene: {scene} Time: {h_seconds_to_timestamp(time_range[0])} -- {h_seconds_to_timestamp(time_range[1])}',
                'transcript': f'Transcript: {transcript}',
                'description': f'{description}',
                'rrf': rrf
                })
                
            
        return jsonify(response)
    else:
        return jsonify("Please send a search term or image")

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '' and request.form['lang']!= '':
            filename = secure_filename(file.filename)
            video_language = request.form['lang']
            if not os.path.exists("./videos"):
                os.makedirs("./videos")
            file.save(os.path.join('./videos/', filename))
            try:
                thread = threading.Thread(target=process_video_to_db, args=[filename,video_language])
                thread.start()
                return jsonify("Video is being processed!")
            except:
                os.remove(f'./videos/{filename}')
                return jsonify("Some error occurred!")
        
@app.route('/logs')
def logs():
    return jsonify(get_logs())

@app.route('/info')
def info():
    response = {}
    response['video_number'] = len(get_video_list()['ids'])
    response['log'] = get_latest_log() 
    return jsonify(response)

@app.route('/status')
def status():
    threads = threading.enumerate()
    if len(threads) > 3: #if larger than 2, there's a video being processed
        return jsonify("Busy")
    else:
        return jsonify("Available")

if __name__ == "__main__":
    app.run(use_reloader=False,port=5000)