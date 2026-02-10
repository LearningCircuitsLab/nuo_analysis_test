import argparse
import deeplabcut

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", required=True)
parser.add_argument("--videos", required=True)   # comma-separated string
parser.add_argument("--device", default="cuda:1")
parser.add_argument("--destfolder", required=True)
args = parser.parse_args()

config_path = args.config_path
videos = args.videos.split(",")  
device = args.device
destfolder = args.destfolder

deeplabcut.analyze_videos(
    config_path,
    videos,
    videotype="mp4",
    shuffle=1,
    trainingsetindex=0,
    device=device,
    save_as_csv=True,
    destfolder=destfolder
)
