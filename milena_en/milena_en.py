import vosk
import queue
from queue import Empty
import json
import torch
import sounddevice as sd
import time
import webbrowser
from fuzzywuzzy import fuzz
import os
import openai
import numpy as np
import re
import threading
import simpleaudio as sa
import pvporcupine
from pvrecorder import PvRecorder
import random
import config_en
import ctypes
from datetime import datetime, timedelta

class TTSWorker(threading.Thread):
    def __init__(self, model, sample_rate, speaker, put_accent, put_yo, device):
        super().__init__(daemon=True,name="TTSWorker")
        self.model = model
        self.sample_rate = sample_rate
        self.speaker = speaker
        self.put_accent = put_accent
        self.put_yo = put_yo
        self.device = device
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
    
    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            text = item
            try:
                audio = self.model.apply_tts(
                    text=text,
                    speaker=self.speaker,
                    sample_rate=self.sample_rate,
                    put_accent=self.put_accent,
                    put_yo=self.put_yo
                )
                start_silence = np.zeros(int(self.sample_rate * 0.25))
                end_silence = np.zeros(int(self.sample_rate * 0.3))
                audio = np.concatenate([start_silence, audio, end_silence])

                sd.play(audio, self.sample_rate)
                sd.wait()
                time.sleep(0.05)
            except Exception as e:
                print("TTS error:", e)
            finally:
                self.queue.task_done()

    def speak_async(self, text: str):
        if not text:
            return
        self.queue.put(text)

    def stop(self, timeout=2.0):
        try:
            self.queue.put(None)
            self.join(timeout=timeout)
        except Exception:
            pass

    
class TimerManager:
    MAX_SECONDS = 30 * 24 * 3600

    def __init__(self, speak_fn, play_fn):
        self.active_timer = None
        self.lock = threading.Lock()
        self.speak = speak_fn
        self.play = play_fn

    def human_time(self, command: str) -> str:
        if not command:
            return ""
        s = str(command)
        m = re.search(r'((?:\d+|\w+)\s+(?:seconds?|minutes?|hours?))', s)
        if m:
            return m.group(1).strip()
        m = re.search(r'(?:in|after)\s+(.+)', s)
        if m:
            return m.group(1).strip()
        return s.strip()
    
    def _timer_finished_callback(self):
        with self.lock:
            info = self.active_timer
            self.active_timer = None
        if not info:
            return
        
        self.alarm = "sounds_en/alarmv1.wav"
        self.play(self.alarm)
        time.sleep(4)
        reminder_text = (info.get("reminder") or "").strip()
        if reminder_text:
            self.speak(f"Reminder: {reminder_text}")
        else:
            timer_text = info.get('text','')
            formatted_time = self.human_time(timer_text)
            self.speak(f"Timer finished: {formatted_time}")

    
    def set_timer(self, secs : int, command_text: str, reminder_msg: str):
        if secs is None:
            self.speak("I didn't understand the timer duration")
            return False
        if secs <= 0:
            self.speak("Cannot set a timer for zero or negative seconds")
            return False
        if secs > self.MAX_SECONDS:
            self.speak("Timer is too long")
            return False
        
        with self.lock:
            if self.active_timer and self.active_timer.get("timer"):
                try:
                    self.active_timer["timer"].cancel()
                except Exception:
                    pass
                self.active_timer = None

            t = threading.Timer(secs, self._timer_finished_callback)
            t.daemon = True
            end_time = datetime.now() + timedelta(seconds=secs)
            self.active_timer = {"timer": t, "text": command_text, "end": end_time, "secs": secs, "reminder": reminder_msg}
            t.start()

        if reminder_msg:
            self.speak(f"Okay, timer set for {self.human_time(command_text)}. I'll remind you: {reminder_msg}")
        else:
            self.speak(f"Okay, timer set for {self.human_time(command_text)}")
        return True
    
    
    def cancel_timer(self):
        with self.lock:
            if not self.active_timer:
                self.speak("No active timers")
                return False
            try:
                self.active_timer["timer"].cancel()
            except Exception:
                pass
            self.active_timer = None
        self.speak("Okay, timer canceled")
        return True
    

    def tell_remaining_timer(self):
        with self.lock:
            info = self.active_timer
        if not info:
            self.speak("There are no active timers now")
            return
        remaining = int((info["end"] - datetime.now()).total_seconds())
        if remaining < 0:
            remaining = 0
        self.speak(f"Time remaining: {self.human_time(str(remaining) + ' seconds')}.")


class MilenaAssistant:
    def __init__(self):
        #config
        self.wakewordkey = config_en.wakewordkey
        self.wakeword_file = config_en.wakeword_file
        self.model_vosk = config_en.model_vosk
        self.listen_timeout = config_en.listen_timeout
        self.language = config_en.language
        self.model_id = config_en.model_id
        self.sample_rate = config_en.sample_rate
        self.speaker = config_en.speaker
        self.put_accent = config_en.put_accent
        self.put_yo = config_en.put_yo
        self.device = config_en.device
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # modules enable
        self.modules_enable = {
            "greetings": config_en.greetings,
            "how_are_you": config_en.greetings,
            "youtube": config_en.youtube_enable,
            "youtube_search": config_en.youtube_enable,
            "google" : config_en.google_enable,
            "google_search": config_en.google_enable,
            "gpt_search": config_en.gpt_search,
            "reboot": config_en.reboot,
            "shutdown": config_en.shutdown,
            "media_next": config_en.media_next,
            "media_prev": config_en.media_prev,
            "media_ps": config_en.media_ps,
            "set_timer": config_en.timer,
            "cancel_timer": config_en.timer,
        }

        # sounds
        self.doing = "sounds_en/doing.wav"
        self.opening = "sounds_en/opening.wav"
        self.done = "sounds_en/done.wav"
        self.niceday = "sounds_en/niceday.wav"
        self.nicedaylong = "sounds_en/nicedaylong.wav"
        self.nicetomeetyou = "sounds_en/nicetomeetyou.wav"
        self.listening = "sounds_en/listening.wav"
        self.yes_sir = "sounds_en/yes_sir.wav"
        self.usure = "sounds_en/areyousure.wav"
        self.allokthx = "sounds_en/allokthx.wav"
        self.thximok = "sounds_en/thximok.wav"
        self.workinok = "sounds_en/workinok.wav"

        #prepare model tts
        self._prepare_model()

        #TTS worker
        self.tts = TTSWorker(self.model, self.sample_rate, self.speaker, self.put_accent, self.put_yo, self.device)
        self.tts.start()

        #Timer manager
        self.timer_manager = TimerManager(self.speak_async, self.play)

        #rekorder/porcupine and recognizer
        self.porcupine = None
        self.recorder = None

        #vosk
        self.rec = vosk.KaldiRecognizer(self.model_vosk, 16000)

        #priority for exact matching
        self.priority = [
            "youtube_search",
            "google_search",
            "youtube",
            "google",
            "gpt_search",
            "reboot",
            "greetings",
            "how_are_you",
            "media_next",
            "media_prev",
            "media_ps",
            "set_timer",
            "cancel_timer",
        ]

        self.commands = {
            "greetings": {
                "phrases": ["hello", "hi", "hey", "greetings"],
                "action": self.cmd_greetings
            },
            "how_are_you": {
                "phrases": ["how are you", "how are you doing", "how do you feel"],
                "action": self.cmd_how_are_you
            },
            "youtube": {
                "phrases": ["youtube", "open youtube", "please open youtube"],
                "action": self.cmd_youtube
            },
            "google" : {
                "phrases" : ["google", "open google", "please open google"],
                "action" : self.cmd_google
            },
            "youtube_search": {
                "phrases": ["search youtube for", "find on youtube"],
                "action": self.cmd_youtube_search
            },
            "google_search": {
                "phrases": ["search google for", "google", "search for"],
                "action": self.cmd_google_search
            },
            "reboot": {
                "phrases": ["reboot", "restart system", "perform reboot", "restart computer", "restart pc"],
                "action": self.cmd_reboot
            },
            "shutdown": {
                "phrases": ["shutdown", "shut down system", "turn off computer", "power off"],
                "action": self.cmd_shutdown
            },
            "gpt_search": {
                "phrases": ["ask the neural network", "ask chat gpt"],
                "action": None
            },
            "media_next": {
                "phrases": ["next track", "next song"],
                "action": self.cmd_media_next
            },
            "media_prev": {
                "phrases": ["previous track", "previous song"],
                "action": self.cmd_media_prev
            },
            "media_ps": {
                "phrases": ["stop","pause", "pause track", "pause song", "play", "resume"],
                "action": self.cmd_media_ps
            },
            "set_timer": {
                "phrases": ["set timer", "timer for", "in", "remind me in"],
                "action": "set_timer"
            },
            "cancel_timer": {
                "phrases": ["cancel timer", "stop timer", "cancel reminder"],
                "action": "cancel_timer"
            },
        }

        # compiled patterns for quick detection
        self.compiled = {}
        for cmd, info in self.commands.items():
            pattern = r'\b(?:' + '|'.join(re.escape(w) for w in info["phrases"]) + r')\b'
            self.compiled[cmd] = re.compile(pattern, flags=re.IGNORECASE)

    def _prepare_model(self):
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language=self.language,
                                  speaker=self.model_id)
        model.to(self.device)
        self.model = model

    def _sound_path(self, filename):
        return os.path.join(self.base_dir, filename)
    
    def play(self, filename):
        try:
            filepath = self._sound_path(filename)
            wave_obj = sa.WaveObject.from_wave_file(filepath)
            _ = wave_obj.play()
        except Exception as e:
            print("play error:", e)

    def random_doing(self):
        doings = [self.doing, self.done, self.opening]
        return random.choice(doings)

    def random_welcome(self):
        welcomes = [self.niceday, self.nicedaylong, self.nicetomeetyou]
        return random.choice(welcomes)

    def random_yes(self):
        yes_l = [self.listening, self.yes_sir]
        return random.choice(yes_l)

    def random_how_are_you(self):
        hay = [self.allokthx, self.thximok, self.workinok]
        return random.choice(hay)

    def speak_async(self, text: str):
        self.tts.speak_async(text)

    def warmup_audio(self):
        silence = np.zeros(int(self.sample_rate * 0.12))
        sd.play(silence, self.sample_rate)
        sd.wait()

    #commands
    def cmd_greetings(self, *args, **kwargs):
        return "Hello, I'm Milena. How can I help you?"

    def cmd_how_are_you(self, *args, **kwargs):
        self.play(self.random_how_are_you())

    def cmd_youtube(self, *args, **kwargs):
        webbrowser.open("https://www.youtube.com")
        self.play(self.random_doing())

    def cmd_google(self, *args, **kwargs):
        webbrowser.open("https://www.google.com")
        self.play(self.random_doing())

    def cmd_youtube_search(self, command: str, *args, **kwargs):
        res_stripe = command.split()
        result = "+".join(res_stripe[3:])
        webbrowser.open(f"https://www.youtube.com/results?search_query={result}")

    def cmd_google_search(self, command: str, *args, **kwargs):
        res_stripe = command.split()
        result = "+".join(res_stripe[3:])
        webbrowser.open(f"https://www.google.com/search?q={result}")

    def cmd_media_next(self, *args, **kwargs):
        ctypes.windll.user32.keybd_event(0xB0,0,0,0)
        self.play(self.random_doing())

    def cmd_media_prev(self, *args, **kwargs):
        ctypes.windll.user32.keybd_event(0xB1,0,0,0)
        self.play(self.random_doing())

    def cmd_media_ps(self, *args, **kwargs):
        ctypes.windll.user32.keybd_event(0xB3,0,0,0)

    def cmd_reboot(self, *args, **kwargs):
        self.play(self.usure)
        answer = self.listen(timeout=self.listen_timeout)
        if "yes" in answer:
            self.play(self.random_doing())
            print("rebooting")
            os.system("shutdown /r /t 0")
        else:
            self.speak_async("Okay, canceling")

    def cmd_shutdown(self, *args, **kwargs):
        self.play(self.usure)
        answer = self.listen(timeout=self.listen_timeout)
        if "yes" in answer:
            self.play(self.random_doing())
            print("shutting down")
            os.system("shutdown /s /t 0")
        else:
            self.speak_async("Okay, canceling")

    def cmd_gpt(self, text: str, *args, **kwargs):
        client = openai.OpenAI(
            api_key=config_en.gpt_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        response = client.responses.create(
            model="llama-3.3-70b-versatile",
            input=f"{text} .(you are assistant Milena, answer briefly and clearly, without 'hello' at the beginning and write numbers in words)"
        )
        print(response.output_text)
        self.speak_async(response.output_text)

    #all for timer
    def words_to_num(self, text: str, numbers_dict: dict) -> int | None:
        if not text:
            return None
        t = text.lower()
        t = re.sub(r'[^a-z0-9\s\-]', ' ', t)
        m = re.search(r'\d+', t)
        if m:
            return int(m.group(0))
        tokens = t.split()
        total = 0
        current = 0
        any_token = False
        for tok in tokens:
            if tok not in numbers_dict:
                continue
            val = numbers_dict[tok]
            any_token = True
            if val >= 1000:
                current = max(1, current) * val
                total += current
                current = 0
            elif val >= 100:
                current = max(1, current) * val
            else:
                current += val
        total += current
        return total if any_token else None

    def extract_reminder_text(self, command: str) -> str:
        if not command:
            return ""
        s = command.lower()
        s = re.sub(r'^\s*(remind me|set timer|timer for|set a timer)\b[\s,:-]*', '', s, flags=re.IGNORECASE)

        time_patterns = [
            r'in\s+[a-z0-9\-\s]+?\s*(?:seconds?|sec|minutes?|min|hours?|hrs?)\b',
            r'\bhalf an hour\b',
            r'\ban hour and a half\b',
            r'\d+\s*(?:seconds?|sec|minutes?|min|hours?|hrs?)\b',
            r'[a-z\-\s]+?\s*(?:seconds?|sec|minutes?|min|hours?|hrs?)\b'
        ]
        for p in time_patterns:
            s = re.sub(p, ' ', s, flags=re.IGNORECASE)

        s = re.sub(r'\b(please|pls)\b', ' ', s, flags=re.IGNORECASE)
        s = re.sub(r'[,\-\:\–]+', ' ', s)
        s = re.sub(r'\s{2,}', ' ', s).strip()
        return s

    def parse_time_from_text(self, command: str) -> int | None:
        cmd = (command or "").lower()
        if re.search(r'half an hour', cmd):
            return 30 * 60
        if re.search(r'hour and a half', cmd):
            return int(1.5 * 3600)

        m = re.search(r'(\d+)\s*(seconds?|sec|minutes?|min|hours?|hrs?)\b', cmd)
        if m:
            val = int(m.group(1))
            unit = m.group(2)
            if "hour" in unit:
                return val * 3600
            if "min" in unit:
                return val * 60
            return val

        m2 = re.search(r'([a-z\-\s]+?)\s*(seconds?|sec|minutes?|min|hours?|hrs?)\b', cmd)
        if m2:
            words = m2.group(1).strip()
            number = self.words_to_num(words, config_en.ENG_NUMBERS)
            if number is not None:
                unit = m2.group(2)
                if "hour" in unit:
                    return number * 3600
                if "min" in unit:
                    return number * 60
                return number

        m3 = re.search(r'in\s+([a-z\-\s\d]+)\s*(seconds?|minutes?|hours?)\b', cmd)
        if m3:
            words = m3.group(1).strip()
            number = self.words_to_num(words, config_en.ENG_NUMBERS)
            if number is not None:
                unit = m3.group(2)
                if "hour" in unit:
                    return number * 3600
                if "min" in unit:
                    return number * 60
                return number
        return None

    #command matching
    def match_command(self, text: str):
        if not text:
            return None, 0
        text = text.lower()
        found = []
        for cmd, pattern in self.compiled.items():
            if pattern.search(text):
                found.append(cmd)
        if found:
            for p in self.priority:
                if p in found:
                    return p, 100

        # fuzzy fallback
        best_match = None
        best_score = 0
        for cmd_key, cmd_data in self.commands.items():
            for phrase in cmd_data["phrases"]:
                score = fuzz.token_sort_ratio(text, phrase)
                if score > best_score:
                    best_match = cmd_key
                    best_score = score
        return best_match, best_score
    
    #listening
    def listen(self, timeout=None):
        if timeout is None:
            timeout = self.listen_timeout
        q = queue.Queue()

        def callback(indata, frames, time_info, status):
            q.put(bytes(indata))

        start_time = time.time()
        try:
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                                channels=1, callback=callback):
                while time.time() - start_time < timeout:
                    try:
                        data = q.get(timeout=0.5)
                    except Empty:
                        continue

                    if self.rec.AcceptWaveform(data):
                        result = self.rec.Result()
                        text = json.loads(result).get('text', '')
                        if text:
                            return text.lower()
        except Exception as e:
            print("Error mic:", e)
        return ""
    
    #run
    def run(self):
        print(config_en.WELCOME_MESSAGE)
        print("At your service")
        self.play(self.random_welcome())
        self.warmup_audio()

        #porcupine
        self.porcupine = pvporcupine.create(
            access_key=self.wakewordkey,
            keyword_paths=[self.wakeword_file]
        )
        self.recorder = PvRecorder(device_index=0, frame_length=self.porcupine.frame_length)

        try:
            self.recorder.start()
        except Exception as e:
            print("recorder start error:", e)
        
        try:
            while True:
                pcm = self.recorder.read()
                keyword_index = self.porcupine.process(pcm)
                if keyword_index >= 0:
                    print("Yes, sir")
                    self.play(self.random_yes())
                    time.sleep(0.1)
                    command = self.listen(timeout=self.listen_timeout)
                    if command in ["exit", "quit", "stop"]:
                        print("Work completed.")
                        break
                    if not command:
                        continue

                    matched_cmd, score = self.match_command(command)

                    if matched_cmd and score >= 70:
                        print(f"Command: {command} (match {score}%)")
                        action = self.commands.get(matched_cmd, {}).get("action")

                        if matched_cmd == "gpt_search" and matched_cmd in self.modules_enable and self.modules_enable[matched_cmd]:
                            self.play(self.listening)
                            next_command = self.listen(timeout=self.listen_timeout)

                            if next_command:
                                self.cmd_gpt(next_command)

                        elif matched_cmd == "youtube_search" and matched_cmd in self.modules_enable and self.modules_enable[matched_cmd]:
                            self.cmd_youtube_search(command)

                        elif matched_cmd == "google_search" and matched_cmd in self.modules_enable and self.modules_enable[matched_cmd]:
                            self.cmd_google_search(command)

                        elif matched_cmd == "set_timer" and matched_cmd in self.modules_enable and self.modules_enable[matched_cmd]:
                            secs = self.parse_time_from_text(command)

                            reminder_msg = self.extract_reminder_text(command)
                            self.timer_manager.set_timer(secs, command, reminder_msg)

                        elif matched_cmd == "cancel_timer" and matched_cmd in self.modules_enable and self.modules_enable[matched_cmd]:
                            self.timer_manager.cancel_timer()

                        elif matched_cmd in self.modules_enable and self.modules_enable[matched_cmd]:
                            if callable(action):
                                result = action(command)
                                if isinstance(result, str):
                                    self.speak_async(result)
                            else:
                                print(f"Command {matched_cmd} not implemented.")
                                self.speak_async(f"Command {matched_cmd} not implemented.")
                        else:
                            print(f"Probably module {matched_cmd} is disabled")
                            self.speak_async(f"Probably module {matched_cmd} is disabled")
                    else:
                        print(f"You haven't added such functions yet: {command}")
                        self.speak_async(f"You haven't added such functions yet: {command}")
        except KeyboardInterrupt:
            print("stopped by user")
        finally:
            #stopping tts
            try:
                self.tts.queue.join()
            except Exception:
                pass
            self.tts.stop(timeout=2.0)

            # stopping recorder
            if self.recorder:
                try:
                    self.recorder.stop()
                    self.recorder.delete()
                except Exception:
                    pass
            if self.porcupine:
                try:
                    self.porcupine.delete()
                except Exception:
                    pass

if __name__ == "__main__":
    assistant = MilenaAssistant()
    assistant.run()