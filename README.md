[Repo link](https://github.com/agamyo168/Lane-Detection/)

## Contributers:

- Fady Ahmed Mostafa Ali فادي احمد مصطفي على, code: 1700957
- Omar Gamal Hamed Agami عمر جمال حامد عجمي, code: 1700860
- Ziad Mostafa Abd El-Aziz Mostafa زياد مصطفى عبد العزيز مصطفى, code: 1700572

## Prerequistes:
### install needed libraries
- pip install jupyter
- pip install numpy
- pip install matplot
- pip install opencv-python
- pip install moviepy
```
pip install jupyter numpy matplot opencv-python moviepy
```

## To run the script:
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
