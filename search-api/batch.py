import sys
from upload import process_video_to_db

def main():
    args = sys.argv[1:]
    pairs = zip(*[iter(args)]*2)
    for v in pairs:
        video_name = str(v[0])
        video_language = str(v[1])
        if video_name and video_language:
            process_video_to_db(video_name,video_language)
    #if args:
    #    for arg in args:
            

if __name__ == "__main__":
    main()