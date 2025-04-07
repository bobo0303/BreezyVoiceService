
from pydantic import BaseModel
import torch

SPEAKERFOLDER = "common_voice_speakers"
OUTPUTPATH = "/tmp/audio_output/"
TASKPAIR = "/tmp/task_pair.json"
CSV_TMP = "/tmp/csv_folder"  
CSV_HEADER_FORMAT =  ['speaker_prompt_audio_filename', 'speaker', 'speaker_prompt_text_transcription', 'content_to_synthesize', 'output_audio_filename']  

#############################################################################

class ModlePath(BaseModel):
    large_v2: str = "models/large-v2.pt"
    medium: str = "models/medium.pt"
    breezyvoice: str = "MediaTek-Research/BreezyVoice-300M"  # first run will auto download model from huggingface
    
############################################################################## 

""" options for Whisper inference """
OPTIONS = {
    "fp16": torch.cuda.is_available(),
    "language": "zh",
    "task": "transcribe",
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6, # default 0.6 | ours 0.2
}

# The whisper inference max waiting time (if over the time will stop it)
WAITING_TIME = 3

#############################################################################

""" Quality check options """
QUALITY_THRESHOLD = 0.3
QUALITY_PASS_TXT = "keep.txt"
QUALITY_FAIL_TXT = "del.txt"

#############################################################################

