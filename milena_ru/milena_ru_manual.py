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
import random
import config_ru
import ctypes
from datetime import datetime, timedelta

class TTSWorker(threading.Thread):
    def __init__(self, model, sample_rate, speaker, put_accent, put_yo, device):
        super().__init__(daemon=True, name="TTSWorker")
        self.model = model
        self.sample_rate = sample_rate
        self.speaker = speaker
        self.put_accent = put_accent
        self.put_yo = put_yo
        self.device = device
        self.queue = queue.Queue()
    
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

    def __init__(self, speal_fn, play_fn):
        self.active_timer = None
        self.lock = threading.Lock()
        self.speak = speal_fn
        self.play = play_fn

    def human_time(self, command: str) -> str:
        if not command:
            return ""
        s = str(command)
        m = re.search(r'((?:\d+|\w+)\s+(?:секунд\w*|минут\w*|час\w*))', s)
        if m:
            return m.group(1).strip()
        m = re.search(r'(?:на|через)\s+(.+)', s)
        if m:
            return m.group(1).strip()
        return s.strip()
    
    def _timer_finished_callback(self):
        with self.lock:
            info = self.active_timer
            self.active_timer = None
        if not info:
            return
        
        self.alarm = "sounds_ru/alarmv1.wav"
        self.play(self.alarm)
        time.sleep(4)
        reminder_text = (info.get("reminder") or "").strip()
        if reminder_text:
            self.speak(f"Напоминание: {reminder_text}")
        else:
            timer_text = info.get('text','')
            formatted_time = self.human_time(timer_text)
            self.speak(f"Таймер завершён: {formatted_time}")

    def set_timer(self, secs : int, command_text: str, reminder_msg: str):
        if secs is None:
            self.speak("не поняла длительность таймера")
            return False
        if secs <= 0:
            self.speak("нельзя поставить таймер на ноли секунд или меньше")
            return False
        if secs > self.MAX_SECONDS:
            self.speak("слишком долгий таймер")
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
            self.speak(f"хорошо, таймер поставлен на {self.human_time(command_text)}. Напомню: {reminder_msg}")
        else:
            self.speak(f"хорошо, таймер поставлен на {self.human_time(command_text)}")
        return True
    
    def cancel_timer(self):
        with self.lock:
            if not self.active_timer:
                self.speak("активных таймеров нет")
                return False
            try:
                self.active_timer["timer"].cancel()
            except Exception:
                pass
            self.active_timer = None
        self.speak("хорошо, таймер отменён")
        return True

    def tell_remaining_timer(self):
        with self.lock:
            info = self.active_timer
        if not info:
            self.speak("сейчас активных таймеров нет")
            return
        remaining = int((info["end"] - datetime.now()).total_seconds())
        if remaining <0:
            remaining = 0
        mins, secs = divmod(remaining, 60)
        hrs, mins = divmod(mins, 60)
        if hrs:
            self.speak(f"Осталось {hrs} часов {mins} минут {secs} секунд.")
        elif mins:
            self.speak(f"Осталось {mins} минут {secs} секунд.")
        else:
            self.speak(f"Осталось {secs} секунд.")

class MilenaAssistant:
    def __init__(self):
        # Конфиг — все параметры должны быть в config.py
        self.wakewordkey = config_ru.wakewordkey
        self.wakeword_file = config_ru.wakeword_file
        self.model_vosk = config_ru.model_vosk
        self.listen_timeout = config_ru.listen_timeout
        self.language = config_ru.language
        self.model_id = config_ru.model_id
        self.sample_rate = config_ru.sample_rate
        self.speaker = config_ru.speaker
        self.put_accent = config_ru.put_accent
        self.put_yo = config_ru.put_yo
        self.device = config_ru.device
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # modules enable
        self.modules_enable = {
            "greetings" : config_ru.greetings,
            "how_are_you" : config_ru.greetings,
            "youtube" : config_ru.youtube_enable,
            "youtube_search" : config_ru.youtube_enable,
            "google" : config_ru.google_enable,
            "google_search" : config_ru.google_enable,
            "gpt_search" : config_ru.gpt_search,
            "reboot" : config_ru.reboot,
            "shutdown" : config_ru.shutdown,
            "media_next" : config_ru.media_next,
            "media_prev" : config_ru.media_prev,
            "media_ps" : config_ru.media_ps,
            "set_timer": config_ru.timer,
            "cancel_timer" : config_ru.timer,
        }

        # sounds
        self.doing = "sounds_ru/doing.wav"
        self.opening = "sounds_ru/opening.wav"
        self.done = "sounds_ru/done.wav"
        self.niceday = "sounds_ru/niceday.wav"
        self.nicedaylong = "sounds_ru/nicedaylong.wav"
        self.nicetomeetyou = "sounds_ru/nicetomeetyou.wav"
        self.listening = "sounds_ru/listening.wav"
        self.yes_sir = "sounds_ru/yes_sir.wav"
        self.usure = 'sounds_ru/areyousure.wav'
        self.allokthx = "sounds_ru/allokthx.wav"
        self.thximok = "sounds_ru/thximok.wav"
        self.workinok = "sounds_ru/workinok.wav"

        # prepare silero-tts
        try:
            model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                      model='silero_tts',
                                      language=self.language,
                                      speaker=self.model_id)
            model.to(self.device)
            self.model = model
        except Exception as e:
            print("Ошибка загрузки TTS модели:", e)
            self.model = None

        # TTS worker
        self.tts = TTSWorker(self.model, self.sample_rate, self.speaker, self.put_accent, self.put_yo, self.device)
        self.tts.start()

        # Timer manager
        self.timer_manager = TimerManager(self.speak_async, self.play)

        # VOSK recognizer (может быть None — мы всё равно используем ручной ввод)
        try:
            if isinstance(self.model_vosk, vosk.Model):
                self.rec = vosk.KaldiRecognizer(self.model_vosk, 16000)
            else:
                if self.model_vosk and os.path.exists(self.model_vosk):
                    model_obj = vosk.Model(self.model_vosk)
                    self.rec = vosk.KaldiRecognizer(model_obj, 16000)
                else:
                    self.rec = None
        except Exception as e:
            print("Ошибка инициализации VOSK:", e)
            self.rec = None

        # priority
        self.priority = [
            "youtube",
            "google",
            "gpt_search",
            "reboot",
            "greetings",
            "how_are_you",
            "media_next",
            "media_prev",
            "media_ps",
            "youtube_search",
            "google_search",
            "set_timer",
            "cancel_timer",
        ]

        self.commands = {
            "greetings": {
                "phrases": ["привет", "здравствуй", "приветик", "салют"],
                "action": self.cmd_greetings
            },
            "how_are_you" : {
                "phrases" : ["как дела","как делишки","как себя чуствуешь"],
                "action" : self.cmd_how_are_you
            },
            "youtube": {
                "phrases": ["ютуб", "открой ютуб", "пожалуйста открой ютуб"],
                "action": self.cmd_youtube
            },
            "google" : {
                "phrases" : ["гугл", "открой гугл", "пожалуйста открой гугл"],
                "action" : self.cmd_google
            },
            "youtube_search" : {
                "phrases" : ["найди в ютубе", "поищи в ютубе"],
                "action" : self.cmd_youtube_search
            },
            "google_search" : {
                "phrases" : ["найди в интернете","поищи в интернете"],
                "action": self.cmd_google_search
            },
            "reboot": {
                "phrases": ["перезагрузка", "перезагрузи систему", "выполни перезагрузку", "перезагрузи компьютер","перезагрузи пк"],
                "action": self.cmd_reboot
            },
            "shutdown" : {
                "phrases": ["выключение","выключи систему","выключи компьютер"],
                "action": self.cmd_shutdown
            },
            "gpt_search": {
                "phrases": ["спроси у нейросети", "спроси у чата жпт"],
                "action": None
            },
            "media_next" : {
                "phrases" : ["следующий трек", "следующая песня"],
                "action" : self.cmd_media_next
            },
            "media_prev" : {
                "phrases" : ["предыдущий трек", "предыдущая песня"],
                "action" : self.cmd_media_prev
            },
            "media_ps" : {
                "phrases" : ["стоп","пауза","останови трек","останови песню","играй"],
                "action" : self.cmd_media_ps
            },
            "set_timer": {
                "phrases": ["поставь таймер", "таймер на", "через", "напомни через"],
                "action": "set_timer"  # special handling
            },
            "cancel_timer": {
                "phrases": ["отмени таймер", "отмена таймера", "отмени напоминание"],
                "action": "cancel_timer"
            },
        }

        # compiled patterns
        self.compiled = {}
        for cmd, info in self.commands.items():
            pattern = r'\b(?:' + '|'.join(re.escape(w) for w in info["phrases"]) + r')\b'
            self.compiled[cmd] = re.compile(pattern, flags=re.IGNORECASE)

    def _sound_path(self, filename):
        return os.path.join(self.base_dir, filename)
    
    def play(self, filename):
        try:
            filepath = self._sound_path(filename)
            if not os.path.exists(filepath):
                return
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
        if self.tts and self.model:
            self.tts.speak_async(text)
        else:
            print("SPEAK:", text)

    def warmup_audio(self):
        silence = np.zeros(int(self.sample_rate * 0.12))
        try:
            sd.play(silence, self.sample_rate)
            sd.wait()
        except Exception:
            pass

    # commands
    def cmd_greetings(self, *args, **kwargs):
        return "Привет, я Милена. Как могу помочь?"

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
        result = "+".join(res_stripe[3:]) if len(res_stripe) > 3 else "+".join(res_stripe[1:])
        webbrowser.open(f"https://www.youtube.com/results?search_query={result}")

    def cmd_google_search(self, command: str, *args, **kwargs):
        res_stripe = command.split()
        result = "+".join(res_stripe[3:]) if len(res_stripe) > 3 else "+".join(res_stripe[1:])
        webbrowser.open(f"https://www.google.com/search?q={result}")

    def cmd_media_next(self, *args, **kwargs):
        try:
            ctypes.windll.user32.keybd_event(0xB0,0,0,0)
            self.play(self.random_doing())
        except Exception:
            pass

    def cmd_media_prev(self, *args, **kwargs):
        try:
            ctypes.windll.user32.keybd_event(0xB1,0,0,0)
            self.play(self.random_doing())
        except Exception:
            pass

    def cmd_media_ps(self, *args, **kwargs):
        try:
            ctypes.windll.user32.keybd_event(0xB3,0,0,0)
        except Exception:
            pass

    def cmd_reboot(self, *args, **kwargs):
        self.play(self.usure)
        answer = input("Вы уверены (да/нет)?: ").strip().lower()
        if "да" in answer:
            self.play(self.random_doing())
            print("перезагрузка (симуляция)")
            #os.system("shutdown /r /t 0")
        else:
            self.speak_async("Хорошо отменяю")

    def cmd_shutdown(self, *args, **kwargs):
        self.play(self.usure)
        answer = input("Вы уверены (да/нет)?: ").strip().lower()
        if "да" in answer:
            self.play(self.random_doing())
            print("выключение (симуляция)")
            #os.system("shutdown /s /t 0")
        else:
            self.speak_async("Хорошо отменяю")

    def cmd_gpt(self, text: str, *args, **kwargs):
        try:
            client = openai.OpenAI(api_key=config_ru.gpt_api_key)
            response = client.responses.create(
                model=config_ru.gpt_model,
                input=f"{text} .(ты ассистент Милена, отвечай кратко и понятно, без 'привет' в начале и пиши цифры буквами)"
            )
            output_text = getattr(response, "output_text", None) or (response.get('output_text') if isinstance(response, dict) else "")
            print(output_text)
            self.speak_async(output_text)
        except Exception as e:
            print("GPT error:", e)
            self.speak_async("Не удалось получить ответ от нейросети.")

    # timer helpers (words_to_num, extract_reminder_text, parse_time_from_text) — как прежде
    def words_to_num(self, text: str, numbers_dict: dict) -> int | None:
        if not text:
            return None
        t = text.lower()
        t = re.sub(r'[^а-яё0-9\s\-]', ' ', t)
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
        s = re.sub(r'^\s*(напомни(?:те)?|поставь\s+таймер(?:\s+на)?|таймер\s+на|установи\s+таймер)\b[\s,:-]*', '', s, flags=re.IGNORECASE)

        time_patterns = [
            r'через\s+[а-яё0-9\-\s]+?\s*(?:секунд(?:[ае])?|сек\b|минут(?:ы|у)?|минуту|мин|час(?:ов|а)?|ч)\b',
            r'\bпол\s*часа\b',
            r'\bполтора(?:\s*часа)?\b',
            r'\d+\s*(?:секунд(?:[ае])?|сек\b|минут(?:ы|у)?|минуту|мин|час(?:ов|а)?|ч)\b',
            r'[а-яё\-\s]+?\s*(?:секунд(?:[ае])?|сек\b|минут(?:ы|у)?|минуту|мин|час(?:ов|а)?|ч)\b'
        ]
        for p in time_patterns:
            s = re.sub(p, ' ', s, flags=re.IGNORECASE)

        s = re.sub(r'\b(пожалуйста|пожалуйст[а-я]*)\b', ' ', s, flags=re.IGNORECASE)
        s = re.sub(r'[,\-\:\–]+', ' ', s)
        s = re.sub(r'\s{2,}', ' ', s).strip()
        return s

    def parse_time_from_text(self, command: str) -> int | None:
        cmd = (command or "").lower()
        if re.search(r'пол\s*часа', cmd):
            return 30 * 60
        if re.search(r'полтора|полторы', cmd):
            if 'час' in cmd:
                return int(1.5 * 3600)
            return int (1.5 * 60)

        m = re.search(r'(\d+)\s*(секунд|секунды|секунд[ае]|\bсек\b|минут|минуты|минуту|мин|часов|часа|час)\b', cmd)
        if m:
            val = int(m.group(1))
            unit = m.group(2)
            if "час" in unit:
                return val * 3600
            if "мин" in unit:
                return val * 60
            return val

        m2 = re.search(r'([а-яё\-\s]+?)\s*(секунд|секунды|секунд[ае]|\bсек\b|минут|минуты|минуту|мин|часов|часа|час)\b', cmd)
        if m2:
            words = m2.group(1).strip()
            number = self.words_to_num(words, getattr(config_ru, "RUS_NUMBERS", {}))
            if number is not None:
                unit = m2.group(2)
                if "час" in unit:
                    return number * 3600
                if "мин" in unit:
                    return number * 60
                return number

        m3 = re.search(r'через\s+([а-яё\-\s\d]+)\s*(секунд|минут|часов|час)\b', cmd)
        if m3:
            words = m3.group(1).strip()
            number = self.words_to_num(words, getattr(config_ru, "RUS_NUMBERS", {}))
            if number is not None:
                unit = m3.group(2)
                if "час" in unit:
                    return number * 3600
                if "мин" in unit:
                    return number * 60
                return number
        return None

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

        best_match = None
        best_score = 0
        for cmd_key, cmd_data in self.commands.items():
            for phrase in cmd_data["phrases"]:
                score = fuzz.token_sort_ratio(text, phrase)
                if score > best_score:
                    best_match = cmd_key
                    best_score = score
        return best_match, best_score

    def listen(self, timeout=None):
        if timeout is None:
            timeout = self.listen_timeout

        # ручной ввод
        try:
            text = input("Введите команду (или 'выход'): ").strip().lower()
            return text
        except EOFError:
            return ""
        except Exception as e:
            print("manual listen error:", e)
            return ""

    def run(self):
        print(getattr(config_ru, "WELCOME_MESSAGE", "Милена активирована."))
        print("К вашим услугам")
        self.play(self.random_welcome())
        self.warmup_audio()

        # Только ручной ввод — основной цикл
        try:
            while True:
                command = self.listen(timeout=self.listen_timeout)
                if command in ["выход", "quit", "exit"]:
                    print("Работа завершена.")
                    break
                if not command:
                    continue

                matched_cmd, score = self.match_command(command)

                if matched_cmd and score >= 70:
                    print(f"Команда: {command} (совпадение {score}%)")
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
                            try:
                                # если функция принимает аргумент, передаём команду
                                result = action(command)
                            except TypeError:
                                result = action()
                            if isinstance(result, str):
                                self.speak_async(result)
                        else:
                            print(f"Команда {matched_cmd} не реализована.")
                            self.speak_async(f"Команда {matched_cmd} не реализована.")
                    else:
                        print(f"Скорее всего модуль {matched_cmd} отключен")
                        self.speak_async(f"Скорее всего модуль {matched_cmd} отключен")
                else:
                    print(f"Неизвестная команда: {command}")
                    self.speak_async(f"Неизвестная команда: {command}")
        except KeyboardInterrupt:
            print("stopped by user")
        finally:
            try:
                self.tts.queue.join()
            except Exception:
                pass
            self.tts.stop(timeout=2.0)

if __name__ == "__main__":
    assistant = MilenaAssistant()
    assistant.run()
