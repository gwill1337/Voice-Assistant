import vosk
import torch
# modules
greetings = True
gpt_search = False
youtube_enable = True
google_enable = True
reboot = True
shutdown = True
media_next = True
media_prev = True
media_ps = True
timer = True

# keys for API
wakewordkey = ""
gpt_api_key = ""

#path's for wakeword and speech model
wakeword_file = r""
model_vosk = vosk.Model(r"")

#model
listen_timeout = 5
language = 'en'
model_id = 'v3_en'
sample_rate = 48000
speaker = 'en_117' #en_60
put_accent = True
put_yo = False
device = torch.device('cpu')




ENG_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100, "thousand": 1000,

    "a": 1, "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "once": 1, "twice": 2
}


WELCOME_MESSAGE = """
>======>                                                 >=>                >>       >=> >====>    
>=>    >=>                       >=>                     >=>               >>=>      >=> >=>   >=> 
>=>    >=> >> >==>    >=>              >==>       >==> >=>>==>            >> >=>     >=> >=>    >=>
>======>    >=>     >=>  >=>     >=> >>   >=>   >=>      >=>   >====>    >=>  >=>    >=> >=>    >=>
>=>         >=>    >=>    >=>    >=> >>===>>=> >=>       >=>            >=====>>=>   >=> >=>    >=>
>=>         >=>     >=>  >=>     >=> >>         >=>      >=>           >=>      >=>  >=> >=>   >=> 
>=>        >==>       >=>        >=>  >====>      >==>    >=>         >=>        >=> >=> >====>    
                              >==>                                                                 
"""