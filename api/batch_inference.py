import os  
import time  
import argparse  
import pandas as pd  
from datasets import Dataset  
  
from api.single_inference import single_inference, CustomCosyVoice  
from g2pw import G2PWConverter  
from lib.constant import ModlePath, QUALITY_PASS_TXT  
from lib.log_config import setup_sys_logging, setup_generate_logging  
from api.utils import load_file_list, zip_wav_files  
  
# Setup logging  
logger = setup_sys_logging()  
generate_logger = setup_generate_logging()  
  
class AudioGenerate:  
    def __init__(self):  
        """  
        Initialize the AudioGenerate class with default attributes.  
          
        :param model_path: str  
            The path to the model used for speech synthesis.  
        """  
        self.models_path = ModlePath()  
        self.customcosyvoice = CustomCosyVoice(self.models_path.breezyvoice)  
        self.bopomofo_converter = G2PWConverter()  
        self.task_flags: dict[str, bool] = {}  
  
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
        round_count = 0  
        data = pd.read_csv(csv_file)  
        total_audio_num = len(data)  
  
        # Transform pandas DataFrame to HuggingFace Dataset  
        dataset = Dataset.from_pandas(data)  
        dataset = dataset.shuffle(seed=int(time.time() * 1000))  
        self.task_flags[task_id] = True  
  
        def gen_audio(row):  
            if self.task_flags[task_id]:  
                speaker_prompt_audio_path = os.path.join(speaker_prompt_audio_folder, f"{row['speaker_prompt_audio_filename']}.wav")  
                speaker_prompt_text_transcription = row['speaker_prompt_text_transcription']  
                content_to_synthesize = row['content_to_synthesize']  
                output_audio_path = os.path.join(output_path, f"{row['output_audio_filename']}.wav")  
  
                if not os.path.exists(speaker_prompt_audio_path):  
                    logger.error(f"File {speaker_prompt_audio_path} does not exist")  
                    return row  # {"status": "failed", "reason": "file not found"}  
                if not os.path.exists(output_audio_path):  
                    single_inference(speaker_prompt_audio_path, content_to_synthesize, output_audio_path, self.customcosyvoice, self.bopomofo_converter, speaker_prompt_text_transcription)  
                else:  
                    pass  
  
        while True:  
            dataset = dataset.map(gen_audio, num_proc=1)  # num_proc can't set over than 1  
            if not self.task_flags[task_id]:  
                break  
            round_count += 1  
            # generate_logger.info(f" | Task {task_id} finished {round_count} times. | ")  
            if quality_check:  
                passed_list = os.path.join(output_path, QUALITY_PASS_TXT)  
                keep_files = load_file_list(passed_list)  
                if len(keep_files) == total_audio_num:  
                    break  
                # else:  
                # generate_logger.info(f" | Task {task_id} haven't generate all audios. | Next round will be started. | ")  
                time.sleep(0.5)  # to avoid stuck another threading (This isn't a good method. we need improve this in the future.)  
            else:  
                audio_files = [f for f in os.listdir(output_path) if f.endswith('.wav')]  
                if len(audio_files) == total_audio_num:  
                    break  
                else:  
                    generate_logger.info(f" | Task {task_id} haven't generate all audios. | Next round will be started. | (something get wrong) ")  
  
        if self.task_flags[task_id]:  
            generate_logger.info(f" | Task {task_id}: audio generation has been completed. | ")  
            generate_logger.info(f" | Start to ZIP all generated wav files. | ")  
            start = time.time()  
            zip_wav_files(output_path)  
            end = time.time()  
            generate_logger.info(f" | ZIP archive {task_id}.zip completed. zip time: {end-start}| ")  
            self.task_flags[task_id] = None  
        else:  
            self.task_flags[task_id] = None  
            generate_logger.info(f" | Task {task_id}: audio generation has been stopped. | ")  
  
    def stop_task(self, task_id: str) -> None:  
        """  
        Stop a running task.  
          
        :param task_id: str  
            The ID of the task to stop.  
        :rtype: None  
        """  
        if self.task_flags.get(task_id, None):  
            self.task_flags[task_id] = False  
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Batch process audio generation.")  
    parser.add_argument("--csv_file", required=True, help="Path to the CSV file containing input data.")  
    parser.add_argument("--speaker_prompt_audio_folder", required=True, help="Path to the folder containing speaker prompt audio files.")  
    parser.add_argument("--output_audio_folder", required=True, help="Path to the folder where results will be stored.")  
    parser.add_argument("--model_path", type=str, required=False, default="MediaTek-Research/BreezyVoice-300M", help="Specifies the model used for speech synthesis.")  
  
    args = parser.parse_args()  
  
    os.makedirs(args.output_audio_folder, exist_ok=True)  
      
    audio_generator = AudioGenerate(args.model_path)  
    audio_generator.process_batch(  
        csv_file=args.csv_file,  
        speaker_prompt_audio_folder=args.speaker_prompt_audio_folder,  
        output_path=args.output_audio_folder,  
        task_id="batch_task",  
        quality_check=False  # Assuming quality_check is False for the command line execution, can be adjusted if needed  
    )  
    
    