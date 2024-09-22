
import subprocess
import torch
import clip
import data
from databasechroma import add_or_update_video, write_to_log
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from build import create_segments_indexes
import json
from PIL import Image
import cv2
import os
import requests
import io
import csv
from cpd_nonlin import cpd_nonlin
import shutil 
import ollama

def scenes_listing(binary_tags):
    start = 0
    end = 0
    count = 1
    scenes = []
    for i, tag in enumerate(binary_tags):
        end = i
        if tag == 1:
            if end > 0:
                scenes.append([start,end])
                count += 1
            start = i
        if i == len(binary_tags)-1:
            scenes.append([start,end])
    return scenes

def segments_list_to_binary_representation(frames,max_frame):
  bin_rep = [0] * (max_frame)
  for frame in frames:
    if frame <= max_frame:
      bin_rep[frame] = 1
  return bin_rep

"""
def get_mean_features_from_segment(features,segment):
    #get mean features for video
    start = segment[0]
    end =segment[1]
    mean = torch.mean(features[start:end],0)
    mean = torch.reshape(mean, (1,-1))
    return mean

def get_time_for_all_scenes(scenes,keyframes):
    time = []
    for scene in scenes:
        start_time = keyframes[scene[0]]
        end_time = keyframes[scene[1]]
        time.append((start_time,end_time))
    return time
"""
def cpd_auto(K, ncp, vmax, desc_rate=1, **kwargs):
    """Main interface
    
    Detect change points automatically selecting their number
        K       - kernel between each pair of frames in video
        ncp     - maximum ncp
        vmax    - special parameter
    Optional arguments:
        lmin     - minimum segment length
        lmax     - maximum segment length
        desc_rate - rate of descriptor sampling (vmax always corresponds to 1x)

    Note:
        - cps are always calculated in subsampled coordinates irrespective to
            desc_rate
        - lmin and m should be in agreement
    ---
    Returns: (cps, costs)
        cps   - best selected change-points
        costs - costs for 0,1,2,...,m change-points
        
    Memory requirement: ~ (3*N*N + N*ncp)*4 bytes ~= 16 * N^2 bytes
    That is 1,6 Gb for the N=10000.
    """
    m = ncp
    (_, scores) = cpd_nonlin(K, m, backtrack=False, **kwargs)
    
    N = K.shape[0]
    N2 = N*desc_rate  # length of the video before subsampling
    
    penalties = np.zeros(m+1)
    # Prevent division by zero (in case of 0 changes)
    ncp = np.arange(1, m+1)
    penalties[1:] = (vmax*ncp/(2.0*N2))*(np.log(float(N2)/ncp)+1)
    
    costs = scores/float(N) + penalties
    m_best = np.argmin(costs)
    (cps, scores2) = cpd_nonlin(K, m_best, **kwargs)

    return (cps, costs)

def get_image_features_and_time(images):
    all_features = []
    time = []
    with torch.no_grad():
      for im in images:
        time.append(im[1])
        image = Image.fromarray(im[0])
        features = data.model.encode_image(data.preprocess(image).unsqueeze(0).to(data.device))
        all_features.append(features)
    return torch.cat(all_features).cpu().numpy(),time

def get_frame_types(video_name):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_name]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)

def process_video_keyframes(video_name):
    try:
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        if not os.path.exists(f'./tmp/img'):
            os.makedirs('./tmp/img')
        for file in os.listdir('./tmp/img'):
            if os.path.isfile(f'./tmp/img/{file}'):
                os.remove(f'./tmp/img/{file}')
    except OSError:
        print ('Error: Creating directory of data')
    frame_types = get_frame_types(f'./videos/{video_name}')
    i_frames = [x[0] for x in frame_types if x[1]=='I']
    if i_frames:
        cam = cv2.VideoCapture(f'./videos/{video_name}')
        video_fps = cam.get(cv2.CAP_PROP_FPS)
        for frame_no in i_frames:
            cam.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            _, frame = cam.read()
            frame_time = str(int((frame_no*100)/video_fps)).zfill(8)
            outname = f'./tmp/img/frame_{frame_time}.jpg'
            cv2.imwrite(outname, frame)
        cam.release()
    #cv2.destroyAllWindows()
    images = []
    for im in os.listdir(f'./tmp/img/'):
        if os.path.isfile(f'./tmp/img/{im}'):
            image = cv2.imread(f'./tmp/img/{im}')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            time = int(im[im.find("_")+1:im.find(".")])
            images.append((image,time))
    images = sorted(images,key=lambda x: x[1])
    image_features, time = get_image_features_and_time(images)
    return image_features, time

def kts_segmentation_keyframes(image_features,fps,frame_count):
    tags= []
    X = cosine_distances(image_features) #distance matrix
    K = np.dot(X, X.T)
    max_clusters = int((frame_count/fps)/10) #maximum cluster number is one cluster per 10 seconds
    size= np.array(image_features).shape[0]
    vmax = 1.0
    (cps, scores) = cpd_auto(K, max_clusters, vmax)
    #(cps, scores2) = cpd_nonlin(K, int(max_clusters/2))
    tags.append(segments_list_to_binary_representation(list(cps),size))
    #print(list(cps))
    #print(tags)
    return tags

def extract_features_from_text(text):
    text_tokenized = None
    while text_tokenized == None:
        try:
            text_tokenized = clip.tokenize(text).to(data.device)
        except:
            text = text[:-10] #reduces the text lenght until it is within the token limit
            text_tokenized = None
    txt_features = []
    with torch.no_grad():
        txt_features = data.model.encode_text(text_tokenized)
    #return txt_features
    return txt_features.tolist()

def translate(text):
    if text != "":
        try:
            response = data.vicuna_client.chat.completions.create(
                model=data.LLM,
                messages=[
                    {"role": "system", "content": "Translate the following text to English"},
                    {"role": "user", "content": text}
                    ],
                max_tokens=data.MAX_TOKENS
            )
        except:
            return ""
        return response.choices[0].message.content
    else:
        return ""

def convert_video_to_audio_ffmpeg(input_file,output_file):
    subprocess.call(["ffmpeg", "-y", "-i", input_file, output_file], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)

def get_transcript_segments(video, video_language):
    convert_video_to_audio_ffmpeg(video,f"./tmp/__audio__.mp3")
    audio_file = f"./tmp/__audio__.mp3"
    files = {'audio_data': open(audio_file, 'rb')}
    options = {
        'language': video_language,
        'model_size': 'medium',
        'word_timestamps':True
    }
    response = requests.post(f"{data.WHISPER_API_ENDPOINT}/transcribe",files=files,data=options)
    transcript_segments = []
    if response.status_code == 200:
        result = response.json()
        for seg in result:
            if video_language == "en":
                translation = seg['text']
            else:
                translation = translate(seg['text'])

            transcript_segments.append({
                'start':int(seg['start']*100),
                'end':int(seg['end']*100),
                'text':seg['text'],
                'translation':translation
            })
    return transcript_segments

def is_between(period_a,period_b):
    if period_a[0]<=period_b[0] and period_a[1]>=period_b[0]:
        return True
    if period_a[0]==period_b[0] and period_a[1]==period_b[1]:
        return True
    if period_a[0]>=period_b[0] and period_b[1]>=period_a[0]:
        return True
    return False

def match_segmentation(segmentation,transcript_segments):
    for segment in segmentation:
        text = ""
        translation = ""
        for t_seg in transcript_segments:
            if is_between((segment['start'],segment['end']),(t_seg['start'],t_seg['end'])):
                text = f"{text} {t_seg['text']}"
                translation = f"{text} {t_seg['translation']}"
        segment['original transcript'] = text
        segment['translated transcript'] = translation
        #if video_language == "en":
        #    segment['translated transcript'] = segment['original transcript']
        #else:
        #    segment['translated transcript'] = translate(segment['original transcript'])
            #segment['translated transcript'] = segment['original transcript']
    return segmentation

def get_range(zipped, start, end):
    items, i = zip(*zipped)
    s = i.index(start)
    e = i.index(end)
    return items[s:e]

def get_last_video_image_files():
    image_files = []
    image_times = []
    for im in os.listdir(f'./tmp/img/'):
        if os.path.isfile(f'./tmp/img/{im}'):
            time = int(im[im.find("_")+1:im.find(".")])
            image_files.append(im)
            image_times.append(time)
    im_time = zip(image_files,image_times)
    im_time = sorted(im_time, key = lambda x: x[1])
    return im_time

def summarize_descriptions(text):
    try:
        response = data.vicuna_client.chat.completions.create(
            model=data.LLM,
            messages=[
                {"role": "system", "content": "Summarize the following descriptions, remove similar descriptions. Make it one sentence long."},
                {"role": "user", "content": text}
                ],
            max_tokens=data.MAX_TOKENS
        )
        return response.choices[0].message.content
    except:
        return ""
    
    #try:
    #    response = data.vicuna_client.chat(
    #        model=data.LLM,
    #        messages=[
    #        {"role": "system", "content": "Summarize the following descriptions, remove similar descriptions. Make it one sentence long."},
    #        {"role": "user", "content": text}  
    #        ]
    #    )
    #    return response['message']['content']
    #except ollama.ResponseError as e:
    #    print('Error: ',e.error)
    #    return ""

def detect_objects(segmentation,threshold=0.5):
    
    im_time = get_last_video_image_files()
    for seg in segmentation:
        objects = []
        images_list = get_range(im_time,seg['start'],seg['end'])
        images_list = list(map(lambda x: './tmp/img/'+x,images_list))
        for im in images_list:
            new_objects = []
            results = data.yolo_model(im, size=1280)
            results = results.pandas().xyxy[0][['name','confidence']]
            for r in results.values.tolist():
                if r[1]>threshold:
                    new_objects.append(r[0])
            objects.extend(x for x in new_objects if x not in objects)
        if objects:
            seg['objects'] = ', '.join(map(str, objects))
        else:
            seg['objects'] = ''
    return segmentation
    #for seg in segmentation:
    #    seg['objects'] = 'A'
    #return segmentation

def get_description(segmentation):
    im_time = get_last_video_image_files()
    for seg in segmentation:
        descriptions = []
        images_filenames = []
        images_list = get_range(im_time,seg['start'],seg['end'])
        images_list = list(map(lambda x: './tmp/img/'+x,images_list))
        for i,im in enumerate(images_list):
            im_bytes_data = io.BytesIO()
            pil_im = Image.open(im)
            pil_im.save(im_bytes_data, format='JPEG')
            im_bytes_data = im_bytes_data.getvalue()
            response = requests.post(f"{data.LAVIS_API_ENDPOINT}/describe",
                                        files = {
                                            "file": (f"_{i}_.jpg",im_bytes_data)
                                            })
            if response.status_code == 200:
                res = response.json()
                if res['description']:
                    descriptions.append(res['description'])
                    images_filenames.append(im)
            else:
                descriptions.append("")
                images_filenames.append("")
        if descriptions:
            seg['description'] = summarize_descriptions('// '.join(map(str, descriptions)))
            #seg['description'] = ''
            #seg['images-descriptions'] = ''.join(map(str, list(zip(images_filenames,descriptions))))
            im_desc=[]
            for i,_ in enumerate(images_filenames):
                im_desc.append({
                    'image_filename':images_filenames[i],
                    'description':descriptions[i]
                })
                #seg['images-descriptions'] = f'{i}//{images_filenames[i]}//{descriptions[i]}'
            seg['images-descriptions'] = im_desc
        else:
            seg['description'] = ''
            seg['images-descriptions'] = []
    return segmentation

def write_eval_data(video_name,transcript_segments,segmentation):
    if not os.path.exists('./eval'):
        os.makedirs(f'./eval')
    if os.path.exists(f'./eval/{video_name}'):
        for file in os.listdir(f'./eval/{video_name}'):
            if os.path.isfile(f'./eval/{video_name}/{file}'):
                os.remove(f'./eval/{video_name}/{file}')
    else:
        os.makedirs(f'./eval/{video_name}')
    #transcript
    keys = transcript_segments[0].keys()
    with open(f'./eval/{video_name}/transcript.csv', 'w+', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, delimiter='|')
        dict_writer.writeheader()
        dict_writer.writerows(transcript_segments)
    #image descriptions
    descriptions = []
    for seg in segmentation:
        #descriptions.append(seg['images-descriptions'])
        descriptions = descriptions + seg['images-descriptions']
    keys = descriptions[0].keys()
    with open(f'./eval/{video_name}/descriptions.csv', 'w+') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, delimiter='|')
        dict_writer.writeheader()
        dict_writer.writerows(descriptions)
    #copy all image files
    src_dir = './tmp/img/'
    dst_dir = f'./eval/{video_name}'
    for file in os.listdir(src_dir):
        if file.endswith(".jpg"):
            shutil.copy(f'{src_dir}/{file}',dst_dir)

def process_video_to_db(video_name,video_language):
    write_to_log(f"{video_name} is starting segmentation")
    print(f"{video_name} is starting segmentation")
    #get image features
    image_features, time = process_video_keyframes(video_name)
    image_features_torch = torch.FloatTensor(image_features).to(data.device)
    image_features_torch = torch.nn.functional.normalize(image_features_torch,dim=1) ##normalize here!!
    image_features = image_features_torch.cpu().numpy()
    
    #Get segmentation
    cam = cv2.VideoCapture(f'./videos/{video_name}')
    video_fps = cam.get(cv2.CAP_PROP_FPS)
    video_frame_count = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    cam.release()
    #cv2.destroyAllWindows()
    #segmentation_tags = agglomerative_result_keyframes(image_features)
    segmentation_tags = kts_segmentation_keyframes(image_features,video_fps,video_frame_count)[0]
    if segmentation_tags:

        scenes = scenes_listing(segmentation_tags)
        segmentation = []
        scene_image_features = []
        for i, scene in enumerate(scenes):
            #save segments
            start_time = time[scene[0]] #the time is in hundreths of a second , 100 is 1 second
            end_time = time[scene[1]]
            segmentation.append({
                'scene':i,
                'start':start_time,
                'end':end_time
            })
            #save image features
            mean = torch.mean(image_features_torch[scene[0]:scene[1]],0)
            mean = torch.reshape(mean, (1,-1))
            scene_image_features.append(mean.tolist()[0])

        write_to_log(f"{video_name} segmentation done successfully")
        print(f"{video_name} segmentation done successfully")
        #save transcripts
        write_to_log(f"{video_name} is starting transcription")
        print(f"{video_name} is starting transcription")
        transcript_segments = get_transcript_segments(f'./videos/{video_name}',video_language)
        if len(transcript_segments) == 0:
            write_to_log(f"Transcript was not successful. Maybe the Whisper API is not working?")
            print(f"Transcript was not successful. Maybe the Whisper API is not working?")
            os.remove(f'./videos/{video_name}')
        segmentation = match_segmentation(segmentation,transcript_segments)
        scene_transcript_features = []
        for seg in segmentation:
            if seg['translated transcript'] != "":
                scene_transcript_features.append(extract_features_from_text(seg['translated transcript'])[0])
            else:
                scene_transcript_features.append(None)
            
        write_to_log(f"{video_name} transcription done successfully")
        print(f"{video_name} transcription done successfully")
        #save detected objects (YOLO)
        #write_to_log(f"{video_name} is starting object detection")
        #segmentation = detect_objects(segmentation)
        #write_to_log(f"{video_name} object detection done successfully")

        #save secondary description (LAVIS)
        write_to_log(f"{video_name} is starting description")
        print(f"{video_name} is starting description")
        segmentation = get_description(segmentation)
        scene_description_features = []
        for seg in segmentation:
            if seg['description'] != "":
                scene_description_features.append(extract_features_from_text(seg['description'])[0])
            else:
                scene_description_features.append(None)
        write_to_log(f"{video_name} description done successfully")
        print(f"{video_name} description done successfully")
        #Save to DB
        metadata = {
            "video name":video_name,
            "fps":video_fps,
            "frame count":video_frame_count,
            "segmentation":json.dumps(segmentation),
            "image features":json.dumps(scene_image_features),
            "transcript features":json.dumps(scene_transcript_features),
            "description features":json.dumps(scene_description_features),
            #more data
        }
        
        add_or_update_video(video_name,metadata)

        ##Now for the Eval data
        write_eval_data(video_name,transcript_segments,segmentation)

        write_to_log(f"{video_name} was uploaded successfully")
        print(f"{video_name} was uploaded successfully")
        #####
        write_to_log(f"Generating index")
        print(f"Generating index")
        create_segments_indexes(video_name)
        write_to_log(f"{video_name} was processed successfully")
        print(f"{video_name} was processed successfully")
    else:
        write_to_log(f"{video_name} was not processed successfully, video is too short")
        print(f"{video_name} was not processed successfully, video is too short")
        os.remove(f'./videos/{video_name}')