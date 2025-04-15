import os  
from lib.constant import SPEAKERFOLDER  
from lib.log_config import setup_sys_logging  
  
# Setup logging  
logger = setup_sys_logging()  
  
def process_batch_task(csv_file: str, output_path: str, task_id: str, audio_generator, quality_check: bool) -> None:  
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
  
def quality_checking_task(task_id: str, model) -> None:  
    """  
    Perform quality check on the audio files of a specific task.  
      
    :param task_id: str  
        The ID of the task to be checked.  
    :param model: Model  
        An instance of the Model class.  
    :rtype: None  
    :logs: Quality check status and errors.  
    """  
    try:  
        model.quality_check(task_id)  
    except Exception as e:  
        logger.error(f"| task ID {task_id} | Error during quality checking: {e} | ")  
        
        