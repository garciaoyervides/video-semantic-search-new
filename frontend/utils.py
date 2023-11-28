import os
import cv2
import subprocess
import numpy as np
from PIL import Image

def parse_script(script_text):
    commands = []
    text_lines = script_text.splitlines()
    try:
        for line in text_lines:
            parts = line.split()
            index = parts[0]
            time = int(parts[1])
            text = " ".join(parts[2:])
            if index != "Images" and index != "Transcripts":
                return False
            if time < 0:
                return False
            commands.append(
                {
                    "index":index,
                    "time":time,
                    "text":text
                }
            )
        return commands
    except:
        return False

def parse_comment(comment_text):
    return comment_text


def get_video_image_sequence(video_file):
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    video_write = open('./tmp/uploaded.mp4', 'wb') # 
    video_write.write(video_file.getvalue())
    #frame_types =get_frame_types('./tmp/uploaded.mp4')
    #i_frames = [x[0] for x in frame_types if x[1]=='I']
    cam = cv2.VideoCapture('./tmp/uploaded.mp4')
    i_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    numElems  = 5
    idx = np.round(np.linspace(0, i_frames - 1, numElems)).astype(int)
    #print(idx)
    frames = []
    for i in idx:
        cam.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = cam.read()
        frames.append(frame)
    im_seq = np.concatenate(frames, axis=1)
    im_seq = cv2.cvtColor(im_seq, cv2.COLOR_BGR2RGB)
    return Image.fromarray(im_seq)

def get_frame_types(video_name):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_name]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)

