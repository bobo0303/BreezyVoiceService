import os  
import time
from lib.constant import SPEAKERFOLDER
from lib.log_config import setup_sys_logging, setup_whisper_logging

from api.batch_inference import AudioGenerate
from api.single_generate_service import SingleAudioGenerate
from api.whisper_api import Model
  
# Setup logging  
logger = setup_sys_logging()  
whisper_logger = setup_whisper_logging()  
  
def process_batch_task(csv_file: str, output_path: str, task_id: str, quality_check: bool, child) -> None:  
    """  
    Process a batch of audio generation tasks.  
      
    :param csv_file: str  
        Path to the CSV file containing input data.  
    :param output_path: str  
        Path to the folder where results will be stored.  
    :param task_id: str  
        The ID of the task being processed.  
    :param audio_generator: AudioGenerate  
        An instance of the AudioGenerate class.  
    :param quality_check: bool  
        Flag indicating whether to perform quality checking.  
    :rtype: None                                                                                                                                                                                                        
    :logs: Batch processing status and errors.  
    """       
    audio_generator = AudioGenerate(child=child)
    try:  
        audio_generator.process_batch(  
            csv_file=csv_file,  
            speaker_prompt_audio_folder=SPEAKERFOLDER,  
            output_path=output_path,  
            task_id=task_id,  
            quality_check=quality_check,  
        )  
    except Exception as e:  
        logger.error(f"| task ID {task_id} | Error processing batch task: {e} | ")  
  
def quality_checking_task(task_id: str, child) -> None:  
    """  
    Perform quality check on the audio files of a specific task.  
      
    :param task_id: str  
        The ID of the task to be checked.  
    :param model: Model  
        An instance of the Model class.  
    :rtype: None  
    :logs: Quality check status and errors.  
    """  
    model = Model(child=child)

    # Directly load the default model  
    whisper_logger.info(" | ########################################################### | ")  
    default_model = "large_v2"  
    model.load_model(default_model)  
    whisper_logger.info(f" | Default model {default_model} has been loaded successfully. | ")  
    
    # Preheat  
    whisper_logger.info(f" | Start to preheat model. | ")  
    default_audio = "audio/test.wav"  
    start = time.time()  
    for _ in range(5):  
        model.transcribe(default_audio, "en")  
    end = time.time()  
    whisper_logger.info(f" | Preheat model has been completed in {end - start:.2f} seconds. | ")  
    whisper_logger.info(" | ########################################################### | ")  
    
    try:  
        model.quality_check(task_id)  
    except Exception as e:  
        logger.error(f"| task ID {task_id} | Error during quality checking: {e} | ")  
        
        
def process_single_task(task_id: str, child, queue):
    audio_generator = SingleAudioGenerate(child=child, queue=queue)
    try:  
        audio_generator.gen_audio()
    except Exception as e:  
        logger.error(f"| task ID {task_id} | Error processing batch task: {e} | ")  

        