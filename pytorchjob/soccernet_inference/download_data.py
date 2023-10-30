#!/usr/bin/env python 
import SoccerNet,os
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=os.path.join(os.getenv('DATA_DIR'),"SoccerNet"))

# download labels SN v2
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])
# download labels for camera shot
mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train","valid","test"]) 

# download Features reduced with PCA
mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["train","valid","test","challenge"])


# FOR Inference dataset
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=os.path.join(os.getenv('DATA_DIR'),"feature_extract"))

# The following will ask for your password (should have been notified in your email after submitting the NDA)
mySoccerNetDownloader.password = input("Password for videos?:\n")

# download LQ Videos
mySoccerNetDownloader.downloadGames(files=["1.mkv", "2.mkv"], split=["train","valid","test","challenge"])


