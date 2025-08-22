@echo off
echo Installing the required modules...

python -m pip install --upgrade pip
python -m pip install vosk sounddevice torch pvporcupine pvrecorder fuzzywuzzy python-Levenshtein omegaconf numpy groq dotenv openai simpleaudio

echo modules instaled. launch assistant...
python "" #<--- your path for python code like "C:\programms\py_projects\project-aid\assistant-linux\milena_en.py"
pause
