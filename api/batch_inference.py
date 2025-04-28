import os  
import time  
import argparse  
import threading
import pandas as pd  
from typing import Optional  
from datasets import Dataset  
  
from api.single_inference import single_inference, CustomCosyVoice  
from g2pw import G2PWConverter  
from lib.constant import ModelPath, Common, QUALITY_PASS_TXT
from lib.log_config import setup_sys_logging, setup_generate_logging  
from api.utils import load_file_list, zip_wav_files  
  
# Setup logging  
logger = setup_sys_logging()  
generate_logger = setup_generate_logging()  
  
class AudioGenerate:  
    def __init__(self, child):  
        """  
        Initialize the AudioGenerate class with default attributes.  
          
        :param model_path: str  
            The path to the model used for speech synthesis.  
        """  
        self.models_path = ModelPath()  
        self.task_flag: Optional[bool] = None  
        self.child = child
        self.stop_event = threading.Event()  
        self.monitor_thread = threading.Thread(target=self.monitor_pipe)  
        self.monitor_thread.start() 
        
    def monitor_pipe(self):  
        while not self.stop_event.is_set():  
            if self.child.poll(): 
                message = self.child.recv()
                self.pc_conn(message)
            else:  
                time.sleep(0.1) 

    def process_batch(self, csv_file: str, speaker_prompt_audio_folder: str, output_path: str, task_id: str, quality_check: bool) -> None:  
        """  
        Process a batch of audio generation tasks.  
          
        :param csv_file: str  
            Path to the CSV file containing input data.  
        :param speaker_prompt_audio_folder: str  
            Path to the folder containing speaker prompt audio files.  
        :param output_path: str  
            Path to the folder where results will be stored.  
        :param task_id: str  
            The ID of the task being processed.  
        :param quality_check: bool  
            Flag indicating whether to perform quality checking.  
        :rtype: None  
        :logs: Batch processing status and progress.  
        """          
        self.task_flag = True  
        
        data = pd.read_csv(csv_file)  
        total_audio_num = len(data)  
        audio_name = data['output_audio_filename'].apply(lambda x: x if str(x).endswith('.wav') else str(x) + '.wav')  
        audio_set = set(audio_name.tolist())
        
        # Transform pandas DataFrame to HuggingFace Dataset  
        dataset = Dataset.from_pandas(data)  
        dataset = dataset.shuffle(seed=int(time.time() * 1000))  
        
        csv_name = os.path.basename(csv_file)
        generate_logger.info(f" | Start to load audio generate sub process: '{csv_name[:-4]}' | ")
        customcosyvoice = CustomCosyVoice(self.models_path.breezyvoice)  
        bopomofo_converter = G2PWConverter()  
        generate_logger.info(f" | audio generate sub process: '{csv_name[:-4]}' loaded | ")
        
        def gen_audio(row):  
            if self.task_flag:  
                speaker_prompt_audio_path = os.path.join(speaker_prompt_audio_folder, f"{row['speaker_prompt_audio_filename']}.wav")  
                speaker_prompt_text_transcription = row['speaker_prompt_text_transcription']  
                content_to_synthesize = row['content_to_synthesize']  
                output_audio_path = os.path.join(output_path, f"{row['output_audio_filename']}.wav")  
  
                if not os.path.exists(speaker_prompt_audio_path):  
                    logger.error(f"File {speaker_prompt_audio_path} does not exist")  
                    return row  # {"status": "failed", "reason": "file not found"}  
                if not os.path.exists(output_audio_path):  
                    single_inference(speaker_prompt_audio_path, content_to_synthesize, output_audio_path, customcosyvoice, bopomofo_converter, speaker_prompt_text_transcription)  
        try:
            while self.task_flag:
                # Traverse the entire data to generate audio files
                dataset = dataset.map(gen_audio, num_proc=1)  # num_proc can't set over than 1  
                
                if quality_check:  
                    passed_list = os.path.join(output_path, QUALITY_PASS_TXT)  
                    keep_files = load_file_list(passed_list)  
                    keep_files = keep_files.intersection(audio_set)  

                    if len(keep_files) == total_audio_num:  
                        break  
                    time.sleep(1)  
                else:  
                    audio_files = [f for f in os.listdir(output_path) if f.endswith('.wav')]  
                    audio_files  = set(audio_files)
                    audio_files = audio_files.intersection(audio_set)  
                    
                    if len(audio_files) == total_audio_num:  
                        break  
                    else:  
                        if self.task_flag:
                            generate_logger.error(f" | Task {task_id} haven't generate all audios. | Next round will be started. | (something get wrong) ")
                        break
    
            if self.task_flag:  
                generate_logger.info(f" | Task {task_id}: audio generation '{csv_name[:-4]}' has been completed. | ")  
                generate_logger.info(f" | Start to ZIP all generated wav files. | ")  
                start = time.time()  
                zip_wav_files(output_path)  
                end = time.time()  
                generate_logger.info(f" | ZIP archive {task_id}.zip completed. zip time: {end-start}| ")  
                self.task_flag = None  
            else:  
                self.task_flag = None  
                generate_logger.info(f" | Task {task_id}: audio generation '{csv_name[:-4]}' has been stopped. | ")  
        finally:
            self.stop_event.set()  
            self.monitor_thread.join()
        
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
            self.child.send(self.task_flag)
                  
