# Project-AID

![Voice Assistant](https://img.shields.io/badge/Voice-Assistant-FF6B6B?style=for-the-badge) 
![Open Source](https://img.shields.io/badge/Open-Source-28A745?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white) 


**Project-AID** (*Assistant Intelligence Daemon*) this voice assistant named Milena, we've 2 versions and 2 language, with voice detection and manual that's can uses for testing, Russian.ver and English.ver.

## Functions
1. Open or search in google/youtube
2. Timer with reminder
3. Media next/pause/previous
4. Shutdown/Reboot pc
5. Ask gpt


## Usage
1. You need to install model for speech to text (STT) from [vosk](https://alphacephei.com/vosk/models). Download the small model in your language(ru or eng).
2. You need to create wakeword file from [picovoice](https://picovoice.ai/) and copy your API tocken for config.(you can create it with your name)
3. Install all sound files in your ver. language from this repo(files must be in the same folder as assistant)
4. Install all modules for python with pip or from requirements.txt:
   ```
    pip install vosk torch sounddevice fuzzywuzzy python-Levenshtein openai numpy simpleaudio pvporcupine pvrecorder

   ```
5. Clone assistant version that you need or all repo, configure all modules in config_ru.py or config_en.py(About config below, config_ru.py works only with russian letters), and that's it.
6. P.s launch the assistant in venv or with ".bat" file.

## Config
If you need to use AI you need API key from [groq](https://groq.com/), I didn't tested this with other platforms.
```python
import vosk
import torch
# modules
greetings = True #<--- this module for commands like "hello" and "how are you"
gpt_search = False #<--- this module if you want to use AI for any question or smth
youtube_enable = True #<--- youtube and google modules enable voice researching and also open youtube or google 
google_enable = True
reboot = True #<--- enable reboot with voice confirm
shutdown = True #<--- enable shutdown with voice confirm
media_next = True #<--- enable media next module for commands like "next track"
media_prev = True #<--- enable media previous module for commands like "previous track"
media_ps = True #<--- enbale play & pause module for commands like "stop" and "play"
timer = True #<--- enable timer if you says like "set timer 15 seconds" or "remind me in 2 minutes", and it's has reminder if you'll say "remind me in 2 minutes take out the pizza" assistant will remind you "remind, take out the pizza"
#also timer has command "cancel timer" but if you put like 2 timers last timer will replase first.

# keys for API
wakewordkey = ""  #<--- API key from picovoice 
gpt_api_key = ""  #<--- API key from groq

#path's for wakeword and speech model
wakeword_file = r"" #<--- your entire path for wakeford file from picovoice must be like smth that "C:\programms\py_projects\project-aid\assistant-linux\milena.ppn"
model_vosk = vosk.Model(r"") #<--- your entire pathh for model that you downloaded from vosk must be like "C:\programms\py_projects\project-aid\assistant-linux\vosk-model-small-ru-0.22"


#below model setting like language and etc. 
#model 
listen_timeout = 5
language = 'ru'
model_id = 'ru_v3'
sample_rate = 48000
speaker = 'baya'
put_accent = True
put_yo = True
device = torch.device('cpu')

#config file has ENG_NUMBERS i don't want to insert the whole of it because it's big but I'll insert a section it looks like:
ENG_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, ...

#and config has 2 versions WELCOME_MESSAGE that just ASCII with front arrows with text "Project-AID" 
```

## Voice commands
English version:
```python
    "greetings" : ["hello", "hi", "hey", "greetings"],
    "how_are_you": ["how are you", "how are you doing", "how do you feel"],
    "youtube": ["youtube", "open youtube", "please open youtube"],
    "google" : ["google", "open google", "please open google"],
    "youtube_search" : ["search youtube for", "find on youtube"],
    "google_search": ["search the internet for", "google", "search for"],
    "reboot": ["reboot", "restart system", "perform reboot", "restart computer", "restart pc"],
    "shutdown": ["shutdown", "shut down system", "turn off computer", "power off"],
    "gpt_search": ["ask the neural network", "ask chat gpt"],
    "media_next": ["next track", "next song"],
    "media_prev":  ["previous track", "previous song"],
    "media_ps": ["stop", "pause", "pause track", "pause song", "play", "resume"],
    "set_timer": ["set timer", "timer for", "in", "remind me in"],
    "cancel_timer": ["cancel timer", "stop timer", "cancel reminder"]
```
Russian version
```python
    "greetings": ["привет", "здравствуй", "приветик", "салют"],
    "how_are_you" : ["как дела","как делишки","как себя чуствуешь"],
    "youtube": ["ютуб", "открой ютуб", "пожалуйста открой ютуб"],
    "google": ["гугл", "открой гугл", "пожалуйста открой гугл"],
    "youtube_search" : ["найди в ютубе", "поищи в ютубе"],
    "google_search" : ["найди в интернете","поищи в интернете"],
    "reboot": ["перезагрузка", "перезагрузи систему", "выполни перезагрузку", "перезагрузи компьютер","перезагрузи пк"]
    "shutdown" : ["выключение","выключи систему","выключи компьютер"],
    "gpt_search": ["спроси у нейросети", "спроси у чата жпт"],
    "media_next" : ["следующий трек", "следующая песня"],
    "media_prev" : ["предыдущий трек", "предыдущая песня"],
    "media_ps" : ["стоп","пауза","останови трек","останови песню","играй"],
    "set_timer": ["поставь таймер", "таймер на", "через", "напомни через"],
    "cancel_timer": ["отмени таймер", "отмена таймера", "отмени напоминание"],
```
