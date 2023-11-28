import json
from databasechroma import write_to_log, get_video, add_segment,delete_all_segments_from_video

#indexes correspond to the frond end indexes, they are:
#Images for image features
#Transcripts for text features
#Description for text features



def create_segments_indexes(video_id):
    delete_all_segments_from_video(video_id)
    video = get_video(video_id)
    ######### IMAGE INDEX
    scenes = json.loads(video['segmentation'])
    image_features = json.loads(video['image features'])
    for i, scene in enumerate(scenes):
        metadata = {
            "video name":video_id,
            "index":"Images",
            "scene":scene['scene'],
            "start":scene['start'],
            "end":scene['end'],
            "transcript":scene['translated transcript'],
            #"objects":scene['objects'],
            "description":scene['description']
        }
        segment_id = f"{video_id} Images {str(i).zfill(4)}"
        if image_features[i] != None and scene['start'] != scene['end']:
            #print(f"Image: {scene['scene']} - {scene['start']} {scene['end']} - {sum(image_features[i])}")
            add_segment(segment_id, image_features[i], metadata)
    write_to_log(f"Video {len(video_id)} has been added to the Image Index")
    
    
    ######### TRANSCRIPT INDEX
    transcript_features = json.loads(video['transcript features'])
    for i, scene in enumerate(scenes):
        metadata = {
            "video name":video_id,
            "index":"Transcripts",
            "scene":scene['scene'],
            "start":scene['start'],
            "end":scene['end'],
            "transcript":scene['translated transcript'],
            #"objects":scene['objects'],
            "description":scene['description']
        }
        segment_id = f"{video_id} Transcripts {str(i).zfill(4)}"
        if transcript_features[i] != None and scene['start'] != scene['end']:
            #print(f"Transcript: {scene['scene']} - {scene['start']} {scene['end']} - {sum(transcript_features[i])}")
            add_segment(segment_id, transcript_features[i], metadata)
    write_to_log(f"Video {len(video_id)} has been added to the Transcripts Index")

    ######### DESCRIPTION INDEX
    description_features = json.loads(video['description features'])
    for i, scene in enumerate(scenes):
        metadata = {
            "video name":video_id,
            "index":"Descriptions",
            "scene":scene['scene'],
            "start":scene['start'],
            "end":scene['end'],
            "transcript":scene['translated transcript'],
            #"objects":scene['objects'],
            "description":scene['description']
        }
        segment_id = f"{video_id} Descriptions {str(i).zfill(4)}"
        if description_features[i] != None and scene['start'] != scene['end']:
            #print(f"Description: {scene['scene']} - {scene['start']} {scene['end']} - {sum(description_features[i])}")
            add_segment(segment_id, description_features[i], metadata)
    write_to_log(f"Video {len(video_id)} has been added to the Transcripts Index")