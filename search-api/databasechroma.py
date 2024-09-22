import io
import json
import numpy as np
from datetime import datetime
from data import videos_collection
from data import segments_collection
from data import logs_collection

MAX_LOG_SIZE = 100

def delete_all_segments_from_video(video_name):
    segments = segments_collection.get(where={'video name':video_name})
    if len(segments['ids']) != 0:
        #the video already exists
        segments_collection.delete(ids=segments['ids'])

def add_or_update_video(video_name,metadata):
    video = videos_collection.get(ids=[video_name])
    if len(video['ids']) == 0:
        #no video exists
        videos_collection.add(
            embeddings=[[0]],
            metadatas=[metadata],
            ids=[video_name]
        )
    else:
        #the video already exists
        videos_collection.update(
            embeddings=[[0]],
            metadatas=[metadata],
            ids=[video_name]
        )   

def add_segment(segment_id, embeddings, metadata):
    segments_collection.add(
            embeddings=[embeddings],
            metadatas=[metadata],
            ids=[segment_id]
    )

def write_to_log(text):
    id = logs_collection.count()
    dt = datetime.now()
    logs_collection.add(
        embeddings=[[0]],
        metadatas=[{"text": text,"timestamp":str(dt)}],
        ids=[str(id+1)]
    )

def get_logs():
    data = logs_collection.get(include=["metadatas"])
    return data['metadatas']

def get_latest_log():
    data = logs_collection.get(include=["metadatas"])['metadatas'][-1]
    message = f"{data['timestamp']} {data['text']}"
    return message

def get_video_name_from_id(id):
    data = videos_collection.get(ids=[str(id)],include=["metadatas"])
    return data['metadatas'][0]['video name']

def get_video(id):
    data = videos_collection.get(ids=[str(id)],include=["metadatas"])
    return data['metadatas'][0]

'''''
def query_segments(embeddings, index='All', k='10'):
    if index == 'All':
        results = segments_collection.query(
            query_embeddings=embeddings,
            n_results=k,
            include=['metadatas','distances']
        )
    else:
        results = segments_collection.query(
                query_embeddings=embeddings,
                n_results=k,
                where={'index':index},
                include=['metadatas','distances']
            )
    return results
'''''

def query_segments(embeddings, index='Images', k='10'):
    results = segments_collection.query(
            query_embeddings=embeddings,
            n_results=k,
            where={'index':index},
            include=['metadatas','distances']
        )
    return results

def get_video_list():
    data = videos_collection.get(include=["metadatas"])
    return data
def get_text_features_size():
    data = videos_collection.get(include=["metadatas"])
    text_features = json.loads(data['metadatas'][0]['text features'])
    return np.shape(text_features)[1]