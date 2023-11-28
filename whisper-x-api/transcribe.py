import whisperx
import subprocess
import data
#from googletrans import Translator
import copy
import os
import requests
import json

def get_transcription(video_file_name,video_language):
    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    convert_video_to_audio_ffmpeg(f"../search-api/videos/{video_file_name}",f"./tmp/__audio__.mp3")
    original_transcript = []
    audio_file = f"./tmp/__audio__.mp3"
    #audio = whisperx.load_audio(audio_file)
    #result = data.model.transcribe(audio,
    #                               language=video_language)
    files = {'audio_data': open(audio_file, 'rb')}
    options = {
        'language': video_language,
        'model_size': 'medium',
        'word_timestamps':True
    }
    response = requests.post(f"{data.API_ENDPOINT}/transcribe",files=files,data=options)
    if response.status_code == 200:
        result = response.json()
        model_a, metadata = whisperx.load_align_model(language_code=video_language, device=data.device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_file, data.device)
        segments = result_aligned["segments"]
        for seg in segments:
            original_transcript.append(
                dict(
                    start=seg['start']*100, #make it so the time is in hundreds of a second (100 = 1 second),
                    end=seg['end']*100,
                    text=seg['text']
                )
            )
        #original_transcript = copy.deepcopy(transcript)
        #join close segments (only once)
        processed_transcript = []
        if get_average_segment_length(original_transcript) < (4.0*100):
            i = 0
            while i < len(original_transcript):
                segment = original_transcript[i]
                if i+1 < len(original_transcript):
                    next_segment = original_transcript[i+1]
                    if next_segment['start']-segment['end'] < segment['end']-segment['start'] + (1.0*100):
                        processed_transcript.append({
                            'start':segment['start'],
                            'end':next_segment['end'],
                            'text':segment['text'] + " "+ next_segment['text']
                        })
                        i+=1
                else:
                    processed_transcript.append(segment)
                i+=1
        else:
            processed_transcript = original_transcript
        #get translated transcript
        translated_transcript = copy.deepcopy(processed_transcript)
        #translator = Translator()
        for segment in translated_transcript:
            #t = translator.translate(segment['text'])
            #segment['text'] = t.text
            segment['text'] = translate(segment['text'])
        return processed_transcript, translated_transcript
    else:
        processed_transcript = []
        translated_transcript = []
        return processed_transcript, translated_transcript

def convert_video_to_audio_ffmpeg(input_file,output_file):
    subprocess.call(["ffmpeg", "-y", "-i", input_file, output_file], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)

def get_average_segment_length(transcript):
    length = 0.0
    for segment in transcript:
        length+=segment['end']-segment['start']
    return length / len(transcript)

def translate(text):
    response = data.client.chat.completions.create(
        model=data.LLM,
        messages=[
            {"role": "system", "content": "Translate the following text to English"},
            {"role": "user", "content": text}
            ],
        max_tokens=data.MAX_TOKENS
    )
    return response.choices[0].message.content

#def get_transcription(file):
