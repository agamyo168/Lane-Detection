[Repo link](https://github.com/agamyo168/Lane-Detection/)

## Contributers:

- Fady Ahmed Mostafa Ali فادي احمد مصطفي على, code: 1700957, Section: 3
- Omar Gamal Hamed Agami عمر جمال حامد عجمي, code: 1700860, Section: 3,
- Ziad Mostafa Abd El-Aziz Mostafa زياد مصطفى عبد العزيز مصطفى, code: 1700572, Section: 2

## Prerequisites:
### install needed libraries
- pip install jupyter
- pip install numpy
- pip install matplot
- pip install opencv-python
- pip install moviepy
- pip install scikit-image
- pip install scikit-learn
```
pip install jupyter numpy matplot opencv-python moviepy scikit-image scikit-learn
```

## To run the script:
### Phase 1:
- open power shell or cmd
- write this command to get output video:

```
.\script './src/challenge_video.mp4' './dst/output.mp4'
```

- to enable debugging mode add -d or --debug to the end of the command

```
.\script './src/challenge_video.mp4' './dst/output.mp4' -d
```

- to run python file directly

```
python .\lane_detection.py './src/challenge_video.mp4' './dst/output.mp4' -d
```
### Phase 2:
- open linux terminal or WSL
- write this command to get output video:
#### HOG
```bash
./car_detection.sh 'src/project_video.mp4' './dst/output.mp4' hog
```
#### YOLO
```bash
./car_detection.sh 'src/project_video.mp4' './dst/output.mp4' yolo
```
## To run the notebook:
When reaching debugging mode, the video will start automatically:
- Press "L" to get next frame.
- Press "Q" to close video.
    
    
