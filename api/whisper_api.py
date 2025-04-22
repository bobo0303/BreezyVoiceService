import os  
import gc  
import time  
import torch  
import whisper  
from opencc import OpenCC  
from typing import Optional  
  
from lib.constant import ModelPath, Common, OPTIONS, QUALITY_PASS_TXT, QUALITY_FAIL_TXT, QUALITY_THRESHOLD, CSV_TMP, OUTPUTPATH  
from lib.log_config import setup_sys_logging, setup_whisper_logging  
from api.utils import calculate_cer, load_csv_data, load_file_list, save_file_list  
  
os.environ["ARGOS_DEVICE_TYPE"] = "cuda"  # Set ARGOS to use CUDA  
cc = OpenCC('s2t')  
  
whisper_logger = setup_whisper_logging()  
logger = setup_sys_logging()  
  
class Model:  
    def __init__(self, child):  
        """  
        Initialize the Model class with default attributes.  
        """  
        self.model: whisper.Whisper = None  
        self.model_version: str = None  
        self.models_path: ModelPath = ModelPath()  
        self.task_flag: Optional[bool] = None  
        self.child = child
  
    def load_model(self, models_name: str) -> None:  
        """  
        Load the specified model based on the model's name.  
          
        :param models_name: str  
            The name of the model to be loaded.  
        :rtype: None  
        :logs: Loading status and time.  
        """  
        start = time.time()  
        try:  
            # Release old model resources  
            self._release_model()  
            self.model_version = models_name  
  
            # Choose model weight  
            if models_name == "large_v2":  
                self.model = whisper.load_model(self.models_path.large_v2)  
            elif models_name == "medium":  
                self.model = whisper.load_model(self.models_path.medium)  
              
            device = "cuda" if torch.cuda.is_available() else "cpu"  
            self.model.to(device)  
            end = time.time()  
            logger.info(f" | Model '{models_name}' loaded in {end - start:.2f} seconds. | ")  
        except Exception as e:  
            self.model_version = None  
            logger.error(f' | load_model() models_name:{models_name} error:{e} | ')  
  
    def _release_model(self) -> None:  
        """  
        Release the resources occupied by the current model.  
          
        :param None: The function does not take any parameters.  
        :rtype: None  
        :logs: Model release status.  
        """  
        if self.model is not None:  
            del self.model  
            gc.collect()  
            self.model = None  
            torch.cuda.empty_cache()  
            logger.info(" | Previous model resources have been released. | ")  
  
    def transcribe(self, audio_file_path: str, ori: str) -> tuple[str, float]:  
        """  
        Perform transcription on the given audio file.  
          
        :param audio_file_path: str  
            The path to the audio file to be transcribed.  
        :param ori: str  
            The original language of the audio.  
        :rtype: tuple  
            A tuple containing the original transcription and inference time.  
        :logs: Inference status and time.  
        """  
        OPTIONS["language"] = ori  
        start = time.time()  
          
        try:  
            result = self.model.transcribe(audio_file_path, **OPTIONS)  
            whisper_logger.debug(result)  
            ori_pred = result['text']  
        except Exception as e:  
            ori_pred = None  
            inference_time = None  
            whisper_logger.error(f' | Transcribe error: {e} | ')  
            whisper_logger.error(f' | Audio file: {audio_file_path} | ')  
            return ori_pred, inference_time  
          
        end = time.time()  
        inference_time = end - start  
        whisper_logger.debug(f" | Inference time {inference_time} seconds. | ")  
          
        return ori_pred, inference_time  
  
    def _process_audio_file(self, audio: str, output_path: str, data: dict, keep_files: set, delete_files: set, passed_list: str, failed_list: str) -> None:  
        """  
        Process a single audio file for quality checking.  
          
        :param audio: str  
            The name of the audio file.  
        :param output_path: str  
            The path to the output directory.  
        :param data: dict  
            The ground truth data for comparison.  
        :param keep_files: set  
            A set of files that passed the quality check.  
        :param delete_files: set  
            A set of files that failed the quality check.  
        :param passed_list: str  
            The path to the file containing the list of files that passed the quality check.  
        :param failed_list: str  
            The path to the file containing the list of files that failed the quality check.  
        :rtype: None  
        :logs: Quality check results.  
        """  
        audio_path = os.path.join(output_path, audio)  
        if not os.path.exists(audio_path):  
            return  
          
        transcription, inference_time = self.transcribe(audio_path, "zh")  
        text = data.get(audio[:-4], 'Content not found')  
        transcription_traditional = cc.convert(transcription).rstrip('。')  
        text = text.rstrip('。')  
          
        whisper_logger.info(f" | GT: {text} | ")  
        whisper_logger.info(f" | ASR: {transcription_traditional} | ")  
          
        if text:  
            cer = calculate_cer(text, transcription_traditional)  
            whisper_logger.info(f" | File: {audio_path}, CER: {cer}, Inference Time: {inference_time:.2f} seconds |")  
              
            if cer < QUALITY_THRESHOLD:  
                keep_files.add(audio)  
            else:  
                if audio not in delete_files:  
                    delete_files.add(audio)  
                if os.path.exists(audio_path):  
                    os.remove(audio_path)  
          
        save_file_list(passed_list, keep_files)  
        save_file_list(failed_list, delete_files)  
  
    def quality_check(self, task_id: str) -> None:  
        """  
        Perform quality check on the audio files of a specific task.  
          
        :param task_id: str  
            The ID of the task to be checked.  
        :rtype: None  
        :logs: Quality check progress and results.  
        """  
        output_path = os.path.join(OUTPUTPATH, task_id)  
        passed_list = os.path.join(output_path, QUALITY_PASS_TXT)  
        failed_list = os.path.join(output_path, QUALITY_FAIL_TXT)  
        csv_file_path = os.path.join(CSV_TMP, task_id + ".csv")  
          
        data = load_csv_data(csv_file_path)  
        total_audio_num = len(data)  
          
        keep_files = load_file_list(passed_list)  
        delete_files = load_file_list(failed_list)  
          
        self.task_flag = True  
          
        while True:  
            if self.task_flag:  
                audio_files = [f for f in os.listdir(output_path) if f.endswith('.wav')]  
                  
                if len(keep_files) == total_audio_num:  
                    whisper_logger.info(f" | Task {task_id}: audio checking has been completed. | ")  
                    self.task_flag = None  
                    break  
            else:  
                whisper_logger.info(f" | Task {task_id}: audio checking has been stopped. | ")  
                self.task_flag = None  
                break  
              
            for audio in audio_files:  
                if self.task_flag:  
                    if audio in keep_files:  
                        continue  
                    self._process_audio_file(audio, output_path, data, keep_files, delete_files, passed_list, failed_list)  
                else:  
                    break  
  
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
            
            