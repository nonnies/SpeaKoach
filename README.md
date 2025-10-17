# Speakoach 
![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)




### Installation 

- _Create new environment in anaconda and install library_

In **_anacoda prompt Shell_** :
```Shell
conda create --name speakoach python=3.10

pip install openai

pip install python-dotenv

python -m pip install dlib-19.22.99-cp310-cp310-win_amd64.whl

pip install numpy==1.24.4

pip install opencv-python-headless==3.4.18.65

pip install opencv-contrib-python==3.4.18.65

pip install playsound sounddevice scipy

pip install streamlit tempfile threading mutagen.mp3
```
### How to Run

In **_anacoda prompt Shell_** :
```Shell
conda activate speakoach
```
run streamlit :
```Shell
streamlit run app.py
```

### Choose camera

- Can change number in _video_src_ to 0,1,2,3,... until you found your camera
_In file **eye_detector.py**_, line 10:
```python
class EyeTracker:
    def __init__(self, video_src=0, predictor_path = "shape_predictor_68_face_landmarks.dat", display=True):

````
> If the program cannot find the _predictor_path_ you can use the **absolute path** instead.


