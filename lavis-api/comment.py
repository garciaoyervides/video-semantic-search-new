import cv2
from PIL import Image
import data
import os
from moviepy.editor import VideoFileClip
import numpy as np

def get_comment(image, prompt="None",nucleus_sampling='False',temperature=0.2,top_percent=0.9):
    
    p_image = data.vis_processors["eval"](image).unsqueeze(0).to(data.device)
    if nucleus_sampling == "True":
        use_nucleus_sampling = True
    else:
        use_nucleus_sampling = False
    if prompt == "None":
        response = data.model.generate({"image": p_image},
                                use_nucleus_sampling=use_nucleus_sampling,
                                top_p=float(top_percent),
                                temperature=float(temperature))
    else:
        response = data.model.generate({"image": p_image, "prompt": prompt},
                                use_nucleus_sampling=use_nucleus_sampling,
                                top_p=float(top_percent),
                                temperature=float(temperature))
    return response[0]

def clip_video(video_path, time, file_name):
    start_time = int(time[0]) / 100
    end_time = int(time[1]) / 100
    video = VideoFileClip(video_path).resize(width=360)
    clip = video.subclip(start_time, end_time)
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    clip.write_videofile(f"./tmp/{file_name}.mp4")
    return f"{file_name}.mp4"

def get_video_image_sequence(video_path,image_file_name):
    cam = cv2.VideoCapture(video_path)
    i_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    numElems  = 5
    idx = np.round(np.linspace(0, i_frames - 1, numElems)).astype(int)
    frames = []
    for i in idx:
        cam.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = cam.read()
        frames.append(frame)
    im_seq = np.concatenate(frames, axis=1)
    im_seq = cv2.cvtColor(im_seq, cv2.COLOR_BGR2RGB)
    Image.fromarray(im_seq).save(f"./tmp/{image_file_name}.jpg")
    return f"{image_file_name}.jpg"