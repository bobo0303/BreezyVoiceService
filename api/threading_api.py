import os
import time
import multiprocessing
from multiprocessing import Process, Queue  
from watchdog.events import FileSystemEventHandler 

from lib.constant import SPEAKERFOLDER
from lib.base_object import BaseResponse
from lib.log_config import setup_sys_logging, setup_whisper_logging

from api.batch_inference import AudioGenerate
from api.custom_audio_generate import SingleAudioGenerate
from api.whisper_api import Model
from api.utils import state_check
  
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
    except KeyboardInterrupt:  
        audio_generator.stop_task()  
        audio_generator.stop_event.set()  
        audio_generator.monitor_thread.join()
    except Exception as e:  
        audio_generator.stop_task()  
        audio_generator.stop_event.set()  
        audio_generator.monitor_thread.join()
        logger.error(f"| task ID {task_id} | Error processing batch task: {e} | ")  
  
def quality_checking_task(task_id: str, child, audio_queue) -> None:  
    """  
    Perform quality check on the audio files of a specific task.  
      
    :param task_id: str  
        The ID of the task to be checked.  
    :param model: Model  
        An instance of the Model class.  
    :rtype: None  
    :logs: Quality check status and errors.  
    """  
    model = Model(child=child, audio_queue=audio_queue)

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
    except KeyboardInterrupt:  
        model.stop_task()  
        model.stop_event.set()  
        model.monitor_thread.join()
    except Exception as e:  
        model.stop_task()  
        model.stop_event.set()  
        model.monitor_thread.join()
        logger.error(f"| task ID {task_id} | Error during quality checking: {e} | ")  
        
        
def process_single_task(task_id: str, child, task_queue, audio_queue):
    audio_generator = SingleAudioGenerate(child=child, task_queue=task_queue, audio_queue=audio_queue)
    try:  
        audio_generator.load_model()
        audio_generator.gen_audio()
    except KeyboardInterrupt: 
        audio_generator.stop_task()  
        audio_generator.stop_event.set()  
        audio_generator.monitor_thread.join()
    except Exception as e:  
        audio_generator.stop_task()  
        audio_generator.stop_event.set()  
        audio_generator.monitor_thread.join()
        logger.error(f"| task ID {task_id} | Error processing batch task: {e} | ")  


def create_and_start_process(task_id, single_generate_state):  
    task_queue = Queue()  
    audio_queue = Queue()  
    generate_parent_conn, generate_child_conn = multiprocessing.Pipe()  
    generate_process = Process(target=process_single_task, args=(task_id, generate_child_conn, task_queue, audio_queue))  
    generate_process.start()  
    single_generate_state["process"] = generate_process  
    single_generate_state["conn"] = generate_parent_conn  
    single_generate_state["task_queue"] = task_queue  
    single_generate_state["audio_queue"] = audio_queue  
    
    return single_generate_state
  
def start_service(task_id, single_generate_state):  
    if single_generate_state["conn"] is not None:  
        state = state_check(task_id, [single_generate_state["conn"]])  
        if isinstance(state, BaseResponse):  
            return state  
        elif not state:  
            single_generate_state = create_and_start_process(task_id, single_generate_state)  
    else:  
        single_generate_state = create_and_start_process(task_id, single_generate_state) 
        
    return single_generate_state

class NewAudioCreate(FileSystemEventHandler):  
    """
    A watchdog event handler that retains only the latest model checkpoint file.
    """
    def __init__(self, audio_queue):  
        super().__init__()  
        self.audio_queue = audio_queue
        
    def on_created(self, event):  
        """
        Called when a new file is created. Cleans up older checkpoints, retains the latest.

        :param event
            The file system event object containing event information.
        :return: None
        """
        if event.is_directory:  
            return None  
        else:  
            if event.src_path.endswith(('.wav')):  
                self.audio_queue.put(os.path.basename(event.src_path))
                # whisper_logger.info(f" | New audio create: {event.src_path} | ")