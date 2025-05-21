import os  
import time  
import queue  
import threading
from typing import Optional 

  
from api.single_inference import single_inference, CustomCosyVoice  
from g2pw import G2PWConverter  
from lib.constant import ModelPath, Common, SPEAKERFOLDER, OUTPUTPATH
from lib.log_config import setup_sys_logging, setup_generate_logging  
  
# Setup logging  
logger = setup_sys_logging()  
generate_logger = setup_generate_logging()  
  
class SingleAudioGenerate:  
    def __init__(self, child, task_queue, audio_queue):  
        """  
        Initialize the AudioGenerate class with default attributes.  
          
        :param model_path: str  
            The path to the model used for speech synthesis.  
        """  
        self.task_flag: Optional[bool] = None  
        self.child = child
        self.task_queue = task_queue 
        self.audio_queue = audio_queue 
        # conn
        self.stop_event = threading.Event()  
        self.monitor_thread = threading.Thread(target=self.monitor_pipe)  
        self.monitor_thread.start() 
        
    def load_model(self):
        # load model
        self.models_path = ModelPath()  
        self.customcosyvoice = CustomCosyVoice(self.models_path.breezyvoice)  
        self.bopomofo_converter = G2PWConverter(num_workers=1)  
        
    def monitor_pipe(self):  
        while not self.stop_event.is_set():  
            if self.child.poll(): 
                message = self.child.recv()
                self.pc_conn(message)
            else:  
                time.sleep(0.1) 
                
    def gen_audio(self):
        output_path = os.path.join(OUTPUTPATH, "tmp")
        os.makedirs(output_path, exist_ok=True)  
        
        self.task_flag = True
        try:
            while self.task_flag:  
                if not self.task_queue.empty():
                    info = self.task_queue.get()
                    speaker_prompt_audio_path = os.path.join(SPEAKERFOLDER, f"{info['speaker']}")  
                    speaker_prompt_text_transcription = info['speaker_text'] 
                    content_to_synthesize = info['text'] 
                    output_audio = os.path.join(output_path, f"{info['output_name']}")  
                    if not os.path.exists(output_audio):  
                        generate_logger.info(f" | Start single generate a audio named {info['output_name']} | speaker: {info['speaker']} | ")
                        single_inference(speaker_prompt_audio_path, content_to_synthesize, output_audio, self.customcosyvoice, self.bopomofo_converter, speaker_prompt_text_transcription)  
                    self.send_audio(info['task_id'], output_audio)
                else:
                    time.sleep(0.1)

        finally:
            self.task_flag = None
            self.stop_event.set()  
            self.monitor_thread.join()
    
    def _remove_existing_task(self, task_id):  
        # Create a temporary list to hold items that are not being removed  
        temp_list = []  
          
        # Transfer items from the original queue to the temporary list  
        while not self.audio_queue.empty():  
            item = self.audio_queue.get()  
            if task_id not in item:  
                temp_list.append(item)  
          
        # Transfer items back from the temporary list to the original queue  
        for item in temp_list:  
            self.audio_queue.put(item)
    
    def send_audio(self, task_id, output_audio):
        self._remove_existing_task(task_id)  
        send_dict = {task_id: output_audio}
        generate_logger.info(f" | task {task_id}: single generate finished put audio {os.path.basename(output_audio)} in audio queue | ")
        self.audio_queue.put(send_dict)
        
    def stop_task(self) -> None:  
        """  
        Stop a running task.  
          
        :param task_id: str  
            The ID of the task to stop.  
        :rtype: None  
        """  
        self.task_flag = False  
        
    def pc_conn(self, message):
        if message == Common.STOP.value:
            self.stop_task()
        elif message == Common.STATE.value:
            try:
                self.child.send(self.task_flag)
            except BrokenPipeError as e:
                pass
            except ConnectionResetError as e:  
                pass
                
                  
