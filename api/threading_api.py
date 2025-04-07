import os
from lib.constant import SPEAKERFOLDER, OUTPUTPATH, QUALITY_PASS_TXT, QUALITY_FAIL_TXT, CSV_TMP
from lib.log_config import setup_sys_logging

logger = setup_sys_logging()  

def process_batch_task(csv_file: str, output_path: str, task_id: str, audio_generator, quality_check):  
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
    
def quality_checking_task(task_id: str, model):  
    try:
        model.quality_check(task_id)
    except Exception as e:
        logger.error(f"| task ID {task_id} | Error during quality checking: {e} | ")