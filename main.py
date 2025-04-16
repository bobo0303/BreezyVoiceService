from fastapi import FastAPI, UploadFile, File, Form  
from fastapi.responses import FileResponse  
import os  
import csv  
import json
import pytz  
import time  
import random
import shutil  
import signal  
import uvicorn  
import threading  
from threading import Thread, Event  
from datetime import datetime  
  
from api.batch_inference import AudioGenerate  
from api.threading_api import process_batch_task, quality_checking_task  
from api.whisper_api import Model  
from api.utils import load_file_list, zip_wav_files  

from lib.constant import OUTPUTPATH, CSV_TMP, CSV_HEADER_FORMAT, SPEAKERFOLDER, SPEAKERS, QUALITY_PASS_TXT, QUALITY_FAIL_TXT  
from lib.base_object import BaseResponse  
from lib.log_config import setup_sys_logging  
  
#############################################################################  
  
# Ensure necessary directories exist  
if not os.path.exists(CSV_TMP):  
    os.makedirs(CSV_TMP)  
if not os.path.exists(OUTPUTPATH):  
    os.makedirs(OUTPUTPATH)  
if not os.path.exists("./logs"):  
    os.mkdir("./logs")  
  
# Setup logging  
logger = setup_sys_logging()  
  
# Configure UTC+8 time  
utc_now = datetime.now(pytz.utc)  
tz = pytz.timezone('Asia/Taipei')  
local_now = utc_now.astimezone(tz)  
  
# Initialize FastAPI app and model  
app = FastAPI()  
model = Model()  
  
#############################################################################  
  
# Log the initialization of BreezyVoice  
logger.info(" | ########################################################### | ")  
logger.info(f" | Start to loading BreezyVoice. | ")  
start = time.time()  
audio_generator = AudioGenerate()  
end = time.time()  
logger.info(f" | BreezyVoice has been loaded successfully in {end - start:.2f} seconds. | ")  
logger.info(" | ########################################################### | ")  
  
# Log any existing tasks  
if os.listdir(OUTPUTPATH):  
    logger.info(f" | The history of alive task | ")  
    for folder in os.listdir(OUTPUTPATH):  
        logger.info(f" | task ID: {folder} | ")  
else:  
    logger.info(f" | no found any previous task alive | ")  
logger.info(" | ########################################################### | ")  
  
#############################################################################  
  
def stop_task(task_id):  
    """   
    Stop the audio generation and model tasks for a given task ID.  
      
    :param task_id: str  
        The ID of the task to stop.  
    :return: BaseResponse  
        A response object indicating the success or failure of the stop operation.  
    """  
    audio_generator.stop_task(task_id)  
    model.stop_task(task_id)  
    waiting_count = 0  
    while True:  
        if audio_generator.task_flags.get(task_id) is None and model.task_flags.get(task_id) is None:  
            break  
        time.sleep(5)  
        waiting_count += 1  
        logger.info(f" | Waiting for task {task_id} to stop... | ")  
        if waiting_count >= 10:  # This isn't a good way to stop the task, but it works. Maybe we can improve it in the future.  
            logger.error(f" | Try to stop Task {task_id} has already 10 times. But nothing happens. | ")  
            return BaseResponse(status="FAILED", message=f" | Failed | Try to stop Task {task_id} has already 10 times. But nothing happens. | ", data=False)  
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
    # Load model  
    default_model = "large_v2"  
    model.load_model(default_model)  # Directly load the default model  
    logger.info(f" | Default model {default_model} has been loaded successfully. | ")  
    # Preheat  
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
    """   
    Get the status and quantity of audio files for a specific task.  
      
    :param task_id: str  
        The ID of the task to check.  
    :return: BaseResponse  
        A response object containing task status and audio file details.  
    """  
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
    """   
    Check the format of a CSV file.  
      
    :param csv_file: UploadFile  
        The CSV file to be checked.  
    :return: BaseResponse  
        A response object indicating whether the CSV format is correct.  
    """  
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
        # Ensure the file is deleted  
        if os.path.exists(csv_file_path):  
            os.remove(csv_file_path)  
            logger.info(f" | Temporary CSV file deleted: {csv_file_path} | ")  
  
@app.post("/batch_generate")  
async def batch_generate(  
    csv_file: UploadFile = File(...),  
    task_id: str = Form(...),  
    quality_check: bool = Form(...),  
):  
    """   
    Start batch audio generation based on a CSV file.  
      
    :param csv_file: UploadFile  
        The CSV file containing batch generation data.  
    :param task_id: str  
        The ID of the task to generate.  
    :param quality_check: bool  
        Flag indicating whether to perform quality checking.  
    :return: BaseResponse  
        A response object indicating the success or failure of the batch generation process.  
    """  
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
    """   
    Cancel a running task.  
      
    :param task_id: str  
        The ID of the task to cancel.  
    :return: BaseResponse  
        A response object indicating the success or failure of the cancellation.  
    """  
    if not audio_generator.task_flags.get(task_id) and not model.task_flags.get(task_id):  
        logger.info(f" | task ID: {task_id} | all process has been stopped. | ")  
        return BaseResponse(status="OK", message=f" | task ID: {task_id} | all process has been stopped. | ", data=True)  
      
    logger.info(f" | Get task ID: {task_id} | ")  
    logger.info(f" | Start to cancel task | ")  
    stop_task(task_id)  
    return BaseResponse(status="OK", message=f" | task ID: {task_id} | all process has been stopped. | ", data=True)  
  
@app.delete("/delete_task")  
def delete_task(task_id: str):  
    """   
    Delete a specific task and its related resources.  
      
    :param task_id: str  
        The ID of the task to delete.  
    :return: BaseResponse  
        A response object indicating the success or failure of the deletion.  
    """  
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
  
@app.get("/get_audio")  
def get_audio(task_id: str):  
    """   
    Get the audio files for a specific task.  
      
    :param task_id: str  
        The ID of the task to get the audio files for.  
    :return: BaseResponse or FileResponse  
        The zipped audio file if successful, otherwise an error response.  
    """  
    output_path = os.path.join(OUTPUTPATH, task_id)  
    zip_file = os.path.join(output_path, task_id + '.zip')  
      
    if not os.path.exists(output_path):  
        logger.error(f" | Please check the task ID is correct | ")  
        return BaseResponse(status="FAILED", message=f" | Please check the task ID is correct | ", data=False)  
      
    if audio_generator.task_flags.get(task_id) or model.task_flags.get(task_id):  
        logger.error(f" | task '{task_id} is still running. please stop it first | ")  
        return BaseResponse(status="FAILED", message=f" | task '{task_id} is still running. please stop it first | ", data=False)  
      
    if os.path.isfile(zip_file):  
        logger.info(f" | zip file found. | ")  
    else:  
        logger.info(f" | '{task_id}.zip' not found. | ")  
        logger.info(f" | Start to zip audios. (It may need some times) | ")  
        start = time.time()  
        try:  
            zip_wav_files(output_path)  
        except Exception as e:  
            logger.error(f"Error zipping files for task {task_id}: {e}")  
            return BaseResponse(status="FAILED", message=f"Error zipping files: {e}", data=False)  
        end = time.time()  
        logger.info(f" | ZIP archive {task_id}.zip completed. zip time: {end-start} | ")  
      
    logger.info(f" | Try to transfer '{task_id}.zip'. | ")  
    return FileResponse(zip_file, media_type='application/zip', filename=f"{task_id}.zip")  
  
@app.put("/zip_audio")  
def zip_audio(task_id: str):  
    """   
    Compress the audio files for a specific task into a zip file.  
      
    :param task_id: str  
        The ID of the task whose audio files will be zipped.  
    :return: BaseResponse  
        A response object indicating the success or failure of the zipping process.  
    """  
    output_path = os.path.join(OUTPUTPATH, task_id)  
    zip_file = os.path.join(output_path, task_id + '.zip')  
      
    if not os.path.exists(output_path):  
        logger.error(f" | Please check the task ID is correct | ")  
        return BaseResponse(status="FAILED", message=f" | Please check the task ID is correct | ", data=False)  
      
    if audio_generator.task_flags.get(task_id) or model.task_flags.get(task_id):  
        logger.error(f" | task '{task_id} is still running. please stop it first | ")  
        return BaseResponse(status="FAILED", message=f" | task '{task_id} is still running. please stop it first | ", data=False)  
      
    if os.path.isfile(zip_file):  
        logger.info(f" | '{task_id}.zip' already excited. | ")  
        logger.info(f" | Start to remove old file and zip again | ")  
        os.remove(zip_file)  
    else:  
        logger.info(f" | Start to zip audios. (It may need some times) | ")  
      
    start = time.time()  
    try:  
        zip_wav_files(output_path)  
    except Exception as e:  
        logger.error(f"Error zipping files for task {task_id}: {e}")  
        return BaseResponse(status="FAILED", message=f"Error zipping files: {e}", data=False)  
    end = time.time()  
    logger.info(f" | ZIP archive {task_id}.zip completed. zip time: {end-start} | ")  
    return BaseResponse(status="OK", message=f" | ZIP archive {task_id}.zip completed. zip time: {end-start} | ", data=True)  

##############################################################################

@app.post("/txt2csv")  
def txt2csv(file: UploadFile = File(...), expansion_ratio: float = 1.0):  
    # Load the text sample to be generated  
    logger.info(f" | Start to load text sample | ")  
      
    if file.filename.endswith(".txt"):  
        texts = file.file.read().decode('utf-8').splitlines()  
        text_num = len(texts)  
    else:  
        logger.info(f" | We only support txt file | ")  
        return BaseResponse(status="FAILED", message=f" | We only support txt file | ", data=False)  
  
    # Load speakers  
    logger.info(f" | Start to load speakers | ")  
    with open(SPEAKERS, 'r') as speakers_file:  
        speakers = json.load(speakers_file)  
        speaker_num = len(speakers)  
  
    # Extend data  
    total_num = int(text_num * expansion_ratio)  
    random.shuffle(texts)  
    extended_texts = (texts * ((total_num // text_num) + 1))[:total_num]  
    random.shuffle(speakers)  
    extended_speakers = (speakers * ((total_num // speaker_num) + 1))[:total_num]  
  
    # Write into CSV format  
    logger.info(f" | Start to write into csv format | ")  
    try:  
        csv_filename = os.path.join(CSV_TMP, file.filename.replace(".txt", ".csv"))  
        with open(csv_filename, 'w', newline='') as csv_file:  
            writer = csv.writer(csv_file)  
            writer.writerow(CSV_HEADER_FORMAT)
            for index, (speaker, text) in enumerate(zip(extended_speakers, extended_texts)):   
                writer.writerows([  
                    [speaker['audio'].replace(".wav",""), speaker['audio'].replace(".wav",""), speaker['sentence'], text, speaker['audio'].rsplit('.', 1)[0] + '_' + str(index+1)]  
                ])  
    except Exception as e:  
        logger.error(f" | Something wrong when writing csv: {e} | ")  
        return BaseResponse(status="FAILED", message=f" | Something wrong when writing csv: {e} | ", data=False)  
  
    logger.info(f" | Generate csv file successful | ")  
    return FileResponse(csv_filename, media_type='application/csv', filename=file.filename.replace(".txt", ".csv"))  
    
  
##############################################################################  
  
# Clean up audio files  
def delete_old_audio_files():  
    """   
    The process of deleting old audio files  
      
    :param None: The function does not take any parameters  
    :rtype: None: The function does not return any value  
    :logs: Deleted old files  
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
    """   
    Schedule a daily cleanup task that removes outdated audio files at midnight.  
      
    :param stop_event: threading.Event  
        A signal used to stop the scheduled task gracefully.  
    """  
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
    """   
    Handle system exit signals (SIGINT, SIGTERM) to gracefully shut down the daily task scheduler.  
    """  
    stop_event.set()  
    task_thread.join()  
    logger.info("Scheduled task has been stopped.")  
    os._exit(0)  
  
signal.signal(signal.SIGINT, handle_exit)  
signal.signal(signal.SIGTERM, handle_exit)  
  
@app.on_event("shutdown")  
def shutdown_event():  
    """   
    FastAPI shutdown event handler to stop background tasks and clean up temporary files.  
    """  
    handle_exit(None, None)  
    stop_event.set()  
    task_thread.join()  
    logger.info("Scheduled task has been stopped.")  
  
if __name__ == "__main__":  
    port = int(os.environ.get("PORT", 80))  
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)  
    
    
    