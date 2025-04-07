from fastapi import FastAPI, UploadFile, File, Form
import os  
import csv
import pytz  
import time  
import shutil
import signal  
import uvicorn  
import threading 
from threading import Thread, Event  
from datetime import datetime  

from api.batch_inference import AudioGenerate  
from api.threading_api import process_batch_task, quality_checking_task
from api.whisper_api import Model  
from api.utils import load_file_list

from lib.constant import OUTPUTPATH, CSV_TMP, CSV_HEADER_FORMAT, SPEAKERFOLDER, QUALITY_PASS_TXT, QUALITY_FAIL_TXT
from lib.base_object import BaseResponse  
from lib.log_config import setup_sys_logging

#############################################################################  

if not os.path.exists(CSV_TMP):  
    os.makedirs(CSV_TMP)  
if not os.path.exists(OUTPUTPATH):
    os.makedirs(OUTPUTPATH)
if not os.path.exists("./logs"):  
    os.mkdir("./logs")  

logger = setup_sys_logging()  

# Configure UTC+8 time  
utc_now = datetime.now(pytz.utc)  
tz = pytz.timezone('Asia/Taipei')  
local_now = utc_now.astimezone(tz)  
  
app = FastAPI()  
model = Model()  

#############################################################################  

logger.info(" | ########################################################### | ")  
logger.info(f" | Start to loading BreezyVoice. | ")  
start = time.time()
audio_generator = AudioGenerate()  
end = time.time()
logger.info(f" | BreezyVoice has been loaded successfully in {end - start:.2f} seconds. | ")  
logger.info(" | ########################################################### | ")  

if os.listdir(OUTPUTPATH):
    logger.info(f" | The history of alive task | ")  
    for folder in os.listdir(OUTPUTPATH):  
        logger.info(f" | task ID: {folder} | ")
else:
    logger.info(f" | no found any previous task alive | ")  
logger.info(" | ########################################################### | ")  

#############################################################################  

def stop_task(task_id):  
    audio_generator.stop_task(task_id)
    model.stop_task(task_id)
    waiting_count = 0
    while True:
        if audio_generator.task_flags.get(task_id) is None and model.task_flags.get(task_id) is None:
            break
        time.sleep(5)
        waiting_count+=1
        logger.info(f" | Waiting for task {task_id} to stop... | ")
        if waiting_count >= 10: # This isn't a good way to stop the task, but it works. maybe we can improve it in the future.
            logger.error(f" | Try to stop Task {task_id} has already 10 times. But not thing happens. | ")
            return BaseResponse(status="FAILED", message=f" | Failed | Try to stop Task {task_id} has already 10 times. But not thing happens. | ", data=False)
        
    logger.info(f" | Task {task_id} all process has been stopped. | ")

#############################################################################  

@app.get("/")  
def HelloWorld(name: str = None):  
    return {"Hello": f"World {name}"}  

#############################################################################  

@app.on_event("startup")  
async def load_default_model_preheat():  
    """  
    The process of loading the default model and preheating on startup.  
  
    This function loads the default model and preheats it by running a few  
    inference operations. This is useful to reduce the initial latency  
    when the model is first used.  
  
    :param None: The function does not take any parameters.  
    :rtype: None: The function does not return any value.  
    :logs: Loading and preheating status and times.  
    """  
    logger.info(f" | Start to loading whisper model. | ")  
    # load model  
    default_model = "large_v2"  
    model.load_model(default_model)  # Directly load the default model  
    logger.info(f" | Default model {default_model} has been loaded successfully. | ")  
    # preheat  
    logger.info(f" | Start to preheat model. | ")  
    default_audio = "audio/test.wav"  
    start = time.time()  
    for _ in range(5):  
        model.transcribe(default_audio, "en")  
    end = time.time()  
      
    logger.info(f" | Preheat model has been completed in {end - start:.2f} seconds. | ")  
    logger.info(" | ########################################################### | ")  
    delete_old_audio_files()  
    
##############################################################################  
    
@app.get("/task_status")
def audio_quantity(task_id: str):    
    output_path = os.path.join(OUTPUTPATH, task_id)  
    passed_list = os.path.join(output_path, QUALITY_PASS_TXT)  
    failed_list = os.path.join(output_path, QUALITY_FAIL_TXT)  
    if not os.path.exists(output_path):  
        logger.error(f" | Please check the task ID is correct | ")
        return BaseResponse(status="FAILED", message=f" | Task not found | ", data=None)
    
    logger.info(f" | Get task ID: {task_id} | ")
    
    audio_files = [f for f in os.listdir(output_path) if f.endswith('.wav')]
    audio_count = len(audio_files)
    
    keep_num = len(load_file_list(passed_list))
    delete_num = len(load_file_list(failed_list))
    unprocessed_num = audio_count - keep_num 
    
    if audio_generator.task_flags.get(task_id) or model.task_flags.get(task_id):
        state = "running"
        logger.info(f" | Task is running. | task ID: {task_id} | ")
    else:
        state = "stopped"
        logger.info(f" | Task is stopped. | task ID: {task_id} | ")
    
    logger.info(f" | Task {task_id} total audio is {audio_count} | keep: {keep_num} | del: {delete_num} | unprocessed: {unprocessed_num} | ")
    
    return_info = {
        "task_id": task_id,
        "state": state,
        "audio_count": audio_count,
        "keep": keep_num,
        "delete": delete_num,
        "unprocessed": unprocessed_num
    }
    
    return BaseResponse(status="OK", message=f" | Task {task_id} is {state}. | total audio: {audio_count} | keep: {keep_num} | del: {delete_num} | unprocessed: {unprocessed_num} | ", data=return_info)

@app.post("/check_csv_format")
async def csv_check(csv_file: UploadFile = File(...)):
    
    csv_file_path = os.path.join(CSV_TMP, csv_file.filename[:-4]+"_checkingfile.csv")
    
    try: 
        with open(csv_file_path, "wb") as buffer:
            buffer.write(await csv_file.read())  
            
        with open(csv_file_path, mode='r', encoding='utf-8-sig') as file:  
            reader = csv.DictReader(file)     
        
            if reader.fieldnames != CSV_HEADER_FORMAT:  
                logger.info(f" | Header is incorrect. Expected: {CSV_HEADER_FORMAT}, Found: {reader.fieldnames} | ")
                return BaseResponse(status="FAILED", message=f" | Header is incorrect. Expected: {CSV_HEADER_FORMAT}, Found: {reader.fieldnames} | ", data=False)

            for row in reader:  
                speakers_list = os.listdir(SPEAKERFOLDER)
                if row['speaker_prompt_audio_filename']+".wav" not in speakers_list:  
                    logger.info(f" | Speaker {row['speaker_prompt_audio_filename']} does not exist | ")
                    return BaseResponse(status="FAILED", message=f" | Speaker {row['speaker_prompt_audio_filename']} does not exist | ", data=False)
                if not all(field in row and row[field] for field in CSV_HEADER_FORMAT):  
                    logger.info(f" | Row is incorrect: {row} | ")
                    return BaseResponse(status="FAILED", message=f" | Row is incorrect: {row} | ", data=False)
            
        logger.info(f" | CSV format is correct. | ")
        return BaseResponse(status="OK", message=f" | format is correct. | ", data=True)
    
    except Exception as e:  
        logger.error(f" | An error occurred: {str(e)} | ")  
        return BaseResponse(status="FAILED", message=f" | An error occurred: {str(e)} | ", data=False) 
    
    finally:  
        # 確保文件被刪除  
        if os.path.exists(csv_file_path):  
            os.remove(csv_file_path)  
            logger.info(f" | Temporary CSV file deleted: {csv_file_path} | ")  
  
@app.post("/batch_generate")  
async def batch_generate(  
    csv_file: UploadFile = File(...),  
    task_id: str = Form(...),
    quality_check: bool = Form(...),
):  
    csv_file_path = os.path.join(CSV_TMP, task_id + ".csv")  
  
    with open(csv_file_path, "wb") as buffer:  
        buffer.write(await csv_file.read())  
    
    output_path = os.path.join(OUTPUTPATH, task_id)  
    
    if os.path.exists(output_path):
        logger.info(f" | task ID: {task_id} already exists. Keep the old audio and continue generate | ")
    else:
        os.makedirs(output_path)
        
    logger.info(f" | Start new task | task ID: {task_id} |")  
    
    generate_process = threading.Thread(target=process_batch_task, args=(csv_file_path, output_path, task_id, audio_generator, quality_check))
    generate_process.start()
    if quality_check:
        quality_process = threading.Thread(target=quality_checking_task, args=(task_id, model))
        quality_process.start()
    else:
        passed_list = os.path.join(output_path, QUALITY_PASS_TXT)  
        failed_list = os.path.join(output_path, QUALITY_FAIL_TXT)  
        if os.path.exists(passed_list):  
            os.remove(passed_list)
        if os.path.exists(failed_list): 
            os.remove(failed_list)
    
    return BaseResponse(status="OK", message=f" | batch generation started. | task ID: {task_id} | ", data=task_id)

@app.post("/cancel_task")  
def cancel_task(task_id: str):  
    if not audio_generator.task_flags.get(task_id) and not model.task_flags.get(task_id):
        logger.info(f" | task ID: {task_id} | all process has been stopped. | ")
        return BaseResponse(status="OK", message=f" | task ID: {task_id} | all process has been stopped. | ", data=True)  

    logger.info(f" | Get task ID: {task_id} | ")
    logger.info(f" | Start to cancel task | ")
    stop_task(task_id)
    return BaseResponse(status="OK", message=f" | task ID: {task_id} | all process has been stopped. | ", data=True)  

@app.delete("/delete_task")
def delete_task(task_id: str):
    if audio_generator.task_flags.get(task_id) or model.task_flags.get(task_id):
        logger.info(f" | Task {task_id} is still running. try to stop the task. | ")
        stop_task(task_id)
        
    output_path = os.path.join(OUTPUTPATH, task_id)  
    if not os.path.exists(output_path):  
        logger.error(f" | Please check the task ID is correct | ")
        return BaseResponse(status="FAILED", message=f" | Task not found | ", data=False)
    else:
        shutil.rmtree(output_path)  
        csv_file_path = os.path.join(CSV_TMP, task_id + ".csv")  
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
        
    logger.info(f" | task ID: {task_id} has been deleted. | ")
    return BaseResponse(status="OK", message=f" | task ID: {task_id} has been deleted. | ", data=True)  


# Clean up audio files  
def delete_old_audio_files():  
    """  
    The process of deleting old audio files  
    :param  
    ----------  
    None: The function does not take any parameters  
    :rtype  
    ----------  
    None: The function does not return any value  
    :logs  
    ----------  
    Deleted old files  
    """  
    current_time = time.time()  
    audio_dir = "./audio"  
    for filename in os.listdir(audio_dir):  
        if filename == "test.wav":  # Skip specific file  
            continue  
        file_path = os.path.join(audio_dir, filename)  
        if os.path.isfile(file_path):  
            file_creation_time = os.path.getctime(file_path)  
            # Delete files older than a day  
            if current_time - file_creation_time > 24 * 60 * 60:  
                os.remove(file_path)  
                logger.info(f"Deleted old file: {file_path}")  
  
# Daily task scheduling  
def schedule_daily_task(stop_event):  
    while not stop_event.is_set():  
        if local_now.hour == 0 and local_now.minute == 0:  
            delete_old_audio_files()  
            time.sleep(60)  # Prevent triggering multiple times within the same minute  
        time.sleep(1)  
  
# Start daily task scheduling  
stop_event = Event()  
task_thread = Thread(target=schedule_daily_task, args=(stop_event,))  
task_thread.start()  
  
# Signal handler for graceful shutdown  
def handle_exit(sig, frame):  
    stop_event.set()  
    task_thread.join()  
    logger.info("Scheduled task has been stopped.")  
    os._exit(0)  
  
signal.signal(signal.SIGINT, handle_exit)  
signal.signal(signal.SIGTERM, handle_exit)  
  
@app.on_event("shutdown")  
def shutdown_event():  
    handle_exit(None, None)  

    stop_event.set()  
    task_thread.join()  
    logger.info("Scheduled task has been stopped.")  

if __name__ == "__main__":  
    port = int(os.environ.get("PORT", 80))  
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)  
    
