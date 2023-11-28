import streamlit as st
import pandas as pd
import requests
import base64
import io
import time
import os
from utils import parse_script, parse_comment, get_video_image_sequence
from config import API_ENDPOINT, LAVIS_API_ENDPOINT
# Define the API endpoint

st.set_page_config(
    page_title="Video Semantic Search Application",
    page_icon="🔍",
)


st.title('Semantic Search Application')
#tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Search a Video","Generate a Video","Script Video","Comment Video","Upload Video","Build Index","Logs"])
tab1, tab2, tab3= st.tabs(["Upload","Search","Comment"])
##VIDEO UPLOAD
with tab1:
    st.caption('Upload a video to the database and regenerate the index to enable search.')
    uploaded_file = st.file_uploader("Choose a video file", help="Max Size 4 GB, use a good descriptive file name", type=['mp4','mkv'])
    video_language = st.text_input('Video Language', help="The ISO 639-1 language code for the video")
    if uploaded_file is not None and video_language:
        if st.button('Upload',key="Upload_Video_Button"):
            with st.spinner('Loading...'):
                bytes_data = uploaded_file.getvalue()
                response = requests.post(f'{API_ENDPOINT}/upload', files =
                                         {"file": (uploaded_file.name,bytes_data)},
                                         data={
                                             "lang": video_language
                                             })
            if response.status_code == 200:
                st.success(response.json())
                with st.status("Server Load", expanded=False) as server_status:
                    videos_number = st.empty()
                    st.text("Latest message:")
                    latest_log = st.empty()
                    info = requests.get(f"{API_ENDPOINT}/info").json()
                    videos_number.text(f"Videos in db: {info['video_number']}")
                    latest_log.text(info['log'])
                    status = "Busy"
                    while status == "Busy":
                        status = requests.get(f"{API_ENDPOINT}/status").json()
                        server_status.update(label="🔴 Busy", state="running", expanded=False)
                        time.sleep(5)
                        info = requests.get(f"{API_ENDPOINT}/info").json()
                        latest_log.text(info['log'])
                    info = requests.get(f"{API_ENDPOINT}/info").json()
                    videos_number.text(f"Videos in db: {info['video_number']}")
                    latest_log.text(info['log'])
                    server_status.update(label="🟢 Available", state="complete", expanded=False)
            else:
                st.error(f'Error uploading file: {response.status_code}')
##VIDEO SEARCH
with tab2:
    st.caption('Searches inside our database and finds the video(s) that are most similar to your search phrase or search image.')
    search_type = st.radio("Search Type",('Text', 'Image'),
                           help="Text: Write down one sentence. Image: Upload one image",
                           key="search_type_radio")
    if search_type == "Text":
        search_term = st.text_input('Search Phrase', help="Longer phrases offer better results")
    if search_type == "Image":
        search_uploaded_file = st.file_uploader("Choose an image file", help="Use a high quality picture", key="Search Upload Image Button", type=['jpg','jpeg','png','bmp'])
        if search_uploaded_file:
            st.image(search_uploaded_file)
    search_results = st.slider('Amount of results', 1, 10, 1)
    #search_expand_treshold = st.slider('Expand Treshold', 0.0, 1.0, 0.05)
    search_index = st.selectbox(
        'Index to query',
        ('Images','Transcripts','Descriptions'),
        help="Images: Matches search phrase with visual data. Transcripts: Matches search phrase with audio spoken in the video.",
        key="index_search")
    if st.button('Search', key="Search Video Button", disabled=
                 ((search_type == "Text" and search_term == "") or
                  (search_type == "Image" and search_uploaded_file is None))):
        with st.spinner('Searching...'):
            if search_type == "Text":
                response = requests.post(f"{API_ENDPOINT}/search",
                                        data={
                                            "text": search_term,
                                            "k": search_results,
                                            "index":search_index
                                            })
            if search_type == "Image":
                search_bytes_data = search_uploaded_file.getvalue()
                response = requests.post(f"{API_ENDPOINT}/search",
                                        files = {
                                            "file": (search_uploaded_file.name,search_bytes_data)
                                            },
                                        data={
                                            "k": search_results,
                                            "index":search_index
                                            })
        if response.status_code == 200:
            with st.spinner('Loading...'):
                data = response.json()
                if not os.path.exists("./tmp"):
                    os.makedirs("./tmp")
                for i,d in enumerate(data):
                    video_decode = base64.b64decode(d['video']) 
                    video_write = open(f'./tmp/segment_{str(i).zfill(3)}.mp4', 'wb')
                    video_write.write(video_decode)
                    video_file = open(f'./tmp/segment_{str(i).zfill(3)}.mp4', 'rb')
                    video_bytes = video_file.read()
                    st.divider()
                    st.write(f"segment_{str(i).zfill(3)}.mp4 Distance: {d['distance']}")
                    if (d['video'] != ""):
                        st.video(video_bytes)
                    if 'identifier' in d:
                        st.caption(f"{d['identifier']}")
                    if 'transcript' in d:
                        st.caption(f"{d['transcript']}")
                    if 'objects' in d:
                        st.caption(f"{d['objects']}")
                    if 'description' in d:
                        st.caption(f"{d['description']}")
        else:
            st.error("Error getting data: {}".format(response.status_code))

##COMMENT VIDEO
with tab3:
    st.caption('Write a natural language input to apply to a video or image of your choosing.')
    comment_input_type = st.radio("Input Type",('Video', 'Image'),
                           help="Upload either image or video.",
                           key="comment_input_type_radio")
    comment_uploaded_image = None
    if comment_input_type == "Video":
        comment_uploaded_video_file = st.file_uploader("Choose a video file", help="You can get a video clip from the search results.", key="Comment Upload Video Button", type=['mp4'])
        if comment_uploaded_video_file:
            image_sequence = get_video_image_sequence(comment_uploaded_video_file)
            #comment_bytes_data = image_sequence
            comment_uploaded_image = st.image(image_sequence)
            comment_bytes_data = io.BytesIO()
            image_sequence.save(comment_bytes_data, format='JPEG')
            comment_bytes_data = comment_bytes_data.getvalue()
    if comment_input_type == "Image":
        comment_uploaded_image_file = st.file_uploader("Choose an image file", help="Use a single picture or a picture composed of a sequence of pictures", key="Comment Upload Image Button", type=['jpg','jpeg','png'])
        if comment_uploaded_image_file:
            comment_uploaded_image = st.image(comment_uploaded_image_file)
            comment_bytes_data = comment_uploaded_image_file.getvalue()
    comment_nucleus_sampling = st.checkbox('Nucleus Sampling', value=False, key="comment_nucleus_sampling_checkbox")
    comment_temperature = st.slider('Temperature', min_value=0.0, max_value=1.0, value=1.0,key="comment_temperature_slider")
    comment_top_percent = st.slider('Top Percent', min_value=0.0, max_value=1.0, value=0.9,key="comment_top_percent_slider")
    comment_video_prompt = st.text_area('Write a prompt for the given input',key="Comment_Video_Input")
    if st.button('Comment',key="Comment_Video_Button_Script", disabled=(comment_video_prompt == "" or not comment_uploaded_image)):
        prompt = parse_comment(comment_video_prompt)
        if prompt:
            with st.spinner('Generating...'):
                response = requests.post(f"{LAVIS_API_ENDPOINT}/comment",
                                        files = {
                                            "file": ("sequence.jpg",comment_bytes_data)
                                            },
                                        data={
                                            "prompt": prompt,
                                            "nucleus_sampling":comment_nucleus_sampling,
                                            "temperature":comment_temperature,
                                            "top_percent":comment_top_percent
                                            })
            if response.status_code == 200:
                data = response.json()
                st.caption(data['comment'])
            else:
                st.error("Error: {}".format(response.status_code))
        else:
            st.error("Error with the script")

_='''
st.header('Chat', divider='rainbow')
st.caption('Searches inside our database and finds the video(s) that are most similar to your search phrase or search image.')
if st.button('Get Videos'):
    video_list = ('A','B','C')
    chat_selected_video = st.selectbox('Available Videos', video_list, placeholder="Select a video to query",)
    if chat_selected_video:
        chat_prompt = st.chat_input("Say something")
'''