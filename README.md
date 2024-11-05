# Video_Synopsis
This code is updated - Original is [NNDam](https://github.com/NNDam) and recode with paper [Surveillance video synopsis framework base on tube set](https://www.sciencedirect.com/science/article/pii/S1047320324000129)
### Require
You need to run Tracking first and save it in a folder. Each frame will save txt conntain Location of each obj
```
frame_1.jpg
frame_1.txt
...
```

### Create background
```
 python extract_background.py --video_path /home1/data/congvu/deepstream-test1/video-synopsis/video_paper.mp4
```
### Video Synopsis
```
python video_synopsis.py  --background_path background.jpg --ROOT /home1/data/congvu/deepstream-test1/synopsis_paper --FPS 25
```
Attention to FPS is FPS of video input

### Experiments
Check in [this](https://youtu.be/vdHDqFTBFA8)
