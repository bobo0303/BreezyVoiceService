from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse  
from fastapi.middleware.cors import CORSMiddleware  
import re  
import io
import os  
import csv  
import json
import pytz  
import time  
import random
import shutil  
import signal  
import hashlib
import zipfile  
import uvicorn  
import subprocess  
import multiprocessing
from multiprocessing import Process, Queue  
from datetime import datetime
from threading import Thread, Event  
from watchdog.observers import Observer  
  
from api.threading_api import process_batch_task, quality_checking_task, start_service, NewAudioCreate
from api.utils import load_file_list, zip_wav_files, state_check, stop_task, split_csv

from lib.constant import OUTPUTPATH, CSV_TMP, LOGPATH, CSV_HEADER_FORMAT, SPEAKERFOLDER, CUSTOMSPEAKERPATH, SPEAKERS, QUALITY_PASS_TXT, QUALITY_FAIL_TXT, OpenAITTSRequest
from lib.base_object import BaseResponse  
from lib.log_config import setup_sys_logging  


#############################################################################  
  
# Initialize FastAPI app and global variable
app = FastAPI()  

app.add_middleware(  
    CORSMiddleware,  
    allow_origins=["*"],
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)  

task_manager = {}
single_generate_state = {
    "process": None,
    "conn": None,
    "task_queue": None,
    "audio_queue": None,
    "usage_list": [],
    "readied_audio": [],
}

# Track usage list history for cleanup
usage_list_history = {}
  
#############################################################################  
  
def initialize_logging_and_directories():  
    # Setup system logging.  
    logger = setup_sys_logging()  
    
    # Configure UTC+8 time zone.  
    utc_now = datetime.now(pytz.utc)  
    tz = pytz.timezone('Asia/Taipei')  
    local_now = utc_now.astimezone(tz)  

    # Ensure necessary directories exist  
    if not os.path.exists(CSV_TMP):  
        os.makedirs(CSV_TMP)  
    if not os.path.exists(OUTPUTPATH):  
        os.makedirs(OUTPUTPATH)  
    if not os.path.exists("./logs"):  
        os.mkdir("./logs")  

    # Log any existing tasks  
    logger.info(" | ########################################################### | ")  
    if os.listdir(OUTPUTPATH):  
        logger.info(f" | The history of alive task | ")  
        for folder in os.listdir(OUTPUTPATH):  
            logger.info(f" | task ID: {folder} | ")  
    else:  
        logger.info(f" | no found any previous task alive | ")  
    logger.info(" | ########################################################### | ")  
  
    return logger, tz, local_now


#############################################################################  
  
@app.get("/")  
def HelloWorld(name: str = None):  
    return {"Hello": f"World {name}"}  
  
#############################################################################   

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

    keep_num = len(load_file_list(passed_list)) if os.path.isfile(passed_list) else 0
    delete_num = len(load_file_list(failed_list)) if os.path.isfile(failed_list) else 0
    unprocessed_num = audio_count - keep_num  
    
    if task_manager.get(task_id):
        generate_state = state_check(task_id, task_manager[task_id]["generate"])
        if isinstance(generate_state, BaseResponse):  
            return generate_state
        quality_state  = state_check(task_id, task_manager[task_id]["quality"])
        if isinstance(quality_state, BaseResponse):  
            return quality_state

        if generate_state or quality_state:  
            state = "running"  
            logger.info(f" | Task is running. | task ID: {task_id} | ")  
        else:  
            state = "stopped"  
            logger.info(f" | Task is stopped. | task ID: {task_id} | ")  
    else:
        state = "stopped"  
        logger.info(f" | Task is stopped. | task ID: {task_id} | ") 
        
    if task_id in single_generate_state["usage_list"]:
        quality_state  = state_check(task_id, [single_generate_state["conn"]])
        single_generate_service_state = "running" if quality_state else "stopped"
    else:
        single_generate_service_state = "stopped"  

    csv_file_path = os.path.join(CSV_TMP, task_id + ".csv")  
    with open(csv_file_path, mode='r', encoding='utf-8') as file:  
        reader = csv.reader(file)  
        next(reader, None)  
        total_audio = sum(1 for row in reader) 
    progress = round(audio_count / total_audio * 100, 2)  

    logger.info(f" | Task {task_id} now audio is {audio_count} | progress: {progress}% | keep: {keep_num} | del: {delete_num} | unprocessed: {unprocessed_num} | ")  
    return_info = {  
        "task_id": task_id,  
        "state": state,  
        "progress": str(progress)+"%",
        "audio_count": audio_count,  
        "keep": keep_num,  
        "delete": delete_num,  
        "unprocessed": unprocessed_num,
        "single_generate_service_state": single_generate_service_state,
    }  
      
    return BaseResponse(status="OK", message=f" | Task {task_id} is {state}. | progress: {progress}% | now audio: {audio_count} | keep: {keep_num} | del: {delete_num} | unprocessed: {unprocessed_num} | ", data=return_info)  
  
@app.post("/custom_speaker_upload")  
async def custom_speaker_upload(task_id: str, file: UploadFile = File(...)):  
    if not file.filename.endswith(".zip"):
        logger.error(f" | Upload speaker should be ZIP file please refer to the sample file first.  | ")  
        return BaseResponse(status="FAILED", message=f" | Upload speaker should be ZIP file please refer to the sample file first. | ", data=False)  
        
    logger.info(f" | Start to create task '{task_id}' custom speakers  | ")  
    file_content = await file.read()  
  
    try:  
        with zipfile.ZipFile(io.BytesIO(file_content)) as zip_file:  
            zip_file_contents = zip_file.namelist()  
            wav_files = [file for file in zip_file_contents if file.endswith('.wav')]  
            json_files = [file for file in zip_file_contents if file.endswith('.json')]  
  
            if len(json_files) != 1:  
                logger.error(f" | no or over two json found. Please check the upload file again. | ")  
                return BaseResponse(status="FAILED", message=f" | no or over two json found. Please check the upload file again | ", data=False)  
            else:  
                json_file_name = json_files[0]  
  
            logger.info(f" | Total {len(wav_files)} audios found | ")  
  
            with zip_file.open(json_file_name) as json_file:  
                json_data = json.load(json_file)  
  
            if len(json_data) != len(wav_files):  
                logger.error(f" | We found json pair '{len(json_data)}' with audio '{len(wav_files)}' not match. Please check the upload file again. | ")  
                return BaseResponse(status="FAILED", message=f" | We found json pair '{len(json_data)}' with audio '{len(wav_files)}' not match. Please check the upload file again | ", data=False)  
  
            for item in json_data:  
                if 'audio' not in item or 'sentence' not in item:  
                    logger.error(f" | Missing 'audio' or 'sentence' in JSON data. line: {item} | ")  
                    return BaseResponse(status="FAILED", message=f" | Missing 'audio' or 'sentence' in JSON data. line: {item} | ", data=False)  
                elif item['audio'] not in wav_files:
                    logger.error(f" | Missing audio '{item['audio']}' | ")
                    return BaseResponse(status="FAILED", message=f" | Missing audio '{item['audio']}' | ", data=False)  
  
            logger.info(f" | Start to save audios | ")  
            for wav_file in wav_files:  
                if wav_file in [item['audio'] for item in json_data]:  
                    wav_output_path = os.path.join(SPEAKERFOLDER, wav_file)  
                    with open(wav_output_path, 'wb') as wav_output_file:  
                        wav_output_file.write(zip_file.read(wav_file))  
                else: 
                    logger.error(f" | found an audio '{wav_file}' not in json pair. Please check the upload file again. | ")
                    return BaseResponse(status="FAILED", message=f" | found an audio '{wav_file}' not in json pair. Please check the upload file again | ", data=False)  
                    
            logger.info(f" | Start to save json | ")  
            os.makedirs(CUSTOMSPEAKERPATH, exist_ok=True)  
            json_output_path = os.path.join(CUSTOMSPEAKERPATH, task_id + ".json")  
            with open(json_output_path, 'wb') as json_output_file:  
                json_output_file.write(zip_file.read(json_file_name))  
            
            logger.info(f" | All process completed. Now task '{task_id}' can use custom speakers | ")  
            return BaseResponse(status="OK", message=f" | All process completed. Now task '{task_id}' can use custom speakers | ", data=True)  
        
    except zipfile.BadZipFile:  
        logger.error(f" | Bad zip file. Please check the upload file again. | ")  
        return BaseResponse(status="FAILED", message=f" | Bad zip file. Please check the upload file again | ", data=False)  
    except json.JSONDecodeError:  
        logger.error(f" | JSON decode error. Please check the upload file again. | ")  
        return BaseResponse(status="FAILED", message=f" | JSON decode error. Please check the upload file again | ", data=False)  
    
  
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
    logger.info(f" | Start to check csv format | ")
    
    if not csv_file.filename.endswith(".csv"):
        logger.error(f" | Upload file should be .csv please check again | ")  
        return BaseResponse(status="FAILED", message=f" | Upload file should be .csv please check again | ", data=False)  
      
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
    num_thread: str = Form("3"),  
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
    
    if task_manager.get(task_id):
        generate_state = state_check(task_id, task_manager[task_id]["generate"])
        if isinstance(generate_state, BaseResponse):  
            return generate_state
        quality_state  = state_check(task_id, task_manager[task_id]["quality"])
        if isinstance(quality_state, BaseResponse):  
            return quality_state
        if generate_state or quality_state:  
            logger.info(f" | Task '{task_id}' is running. | If you want to restart the task, please stop it first. | ")  
            return BaseResponse(status="FAILED", message=f" | Task '{task_id}' is running. | If you want to restart the task, please stop it first. | ", data=False) 
        
    csv_file_path = os.path.join(CSV_TMP, task_id + ".csv")  
    
    with open(csv_file_path, "wb") as buffer:  
        buffer.write(await csv_file.read())  
    
    csv_horcruxes = split_csv(csv_file_path, int(num_thread))
      
    output_path = os.path.join(OUTPUTPATH, task_id)  
      
    if os.path.exists(output_path):  
        logger.info(f" | task ID: {task_id} already exists. Keep the old audio and continue generate | ")  
    else:  
        os.makedirs(output_path)  
        logger.info(f" | Start new task | task ID: {task_id} |")  
        
    generate_parent_conns = []
    for horcrux in csv_horcruxes:
        generate_parent_conn, generate_child_conn = multiprocessing.Pipe()
        generate_process = Process(target=process_batch_task, args=(horcrux, output_path, task_id, quality_check, generate_child_conn))  
        generate_process.start()  
        generate_parent_conns.append(generate_parent_conn)
    
    quality_parent_conns = []
    if quality_check:  
        audio_queue = Queue()
        quality_parent_conn, quality_child_conn = multiprocessing.Pipe()
        quality_process = Process(target=quality_checking_task, args=(task_id, quality_child_conn, audio_queue))  
        quality_process.start()  
        event_handler = NewAudioCreate(audio_queue)
        observer = Observer()  
        observer.schedule(event_handler, path=output_path, recursive=False)
        observer.start()  
    else:  
        quality_parent_conn = None
        observer = None
        passed_list = os.path.join(output_path, QUALITY_PASS_TXT)  
        failed_list = os.path.join(output_path, QUALITY_FAIL_TXT)  
        if os.path.exists(passed_list):  
            os.remove(passed_list)  
        if os.path.exists(failed_list):  
            os.remove(failed_list)  
            
    quality_parent_conns.append(quality_parent_conn)
    
    # add task into task manager 
    task_manager[task_id] = {"generate": generate_parent_conns, "quality": quality_parent_conns, "watchdog": observer} 
    
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
    if task_manager.get(task_id):
        generate_state = state_check(task_id, task_manager[task_id]["generate"])
        if isinstance(generate_state, BaseResponse):  
            return generate_state
        quality_state  = state_check(task_id, task_manager[task_id]["quality"])
        if isinstance(quality_state, BaseResponse):  
            return quality_state
        
        if not generate_state and not quality_state:  
            logger.info(f" | task ID: {task_id} | all process has been stopped. | ")  
            return BaseResponse(status="OK", message=f" | task ID: {task_id} | all process has been stopped. | ", data=True) 
    else:
        logger.info(f" | task '{task_id}' not found | ")  
        return BaseResponse(status="OK", message=f" | task '{task_id}' not found | ", data=False)       

    logger.info(f" | Get task ID: {task_id} | ")  
    logger.info(f" | Start to cancel task | ")  
    stop_info = stop_task(task_id, task_manager[task_id]["generate"], task_manager[task_id]["quality"])  
    if stop_info:
        return stop_info
    try:
        if task_manager[task_id]["watchdog"]:
            task_manager[task_id]["watchdog"].stop()  
            task_manager[task_id]["watchdog"].join()  
    except Exception as e:
        logger.error(f" | Error stopping watchdog: {e} | ")  
        return BaseResponse(status="FAILED", message=f" | Error stopping watchdog: {e} | ", data=False)
    
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
    if task_manager.get(task_id):
        generate_state = state_check(task_id, task_manager[task_id]["generate"])
        if isinstance(generate_state, BaseResponse):  
            return generate_state
        quality_state  = state_check(task_id, task_manager[task_id]["quality"])
        if isinstance(quality_state, BaseResponse):  
            return quality_state
        
        if generate_state or quality_state:  
            logger.info(f" | Task {task_id} is still running. try to stop the task. | ")
            stop_info = stop_task(task_id, task_manager[task_id]["generate"], task_manager[task_id]["quality"])  
            if stop_info:
                return stop_info  
        
    csv_file = os.path.join(CSV_TMP, task_id + ".csv")  
    if os.path.isfile(csv_file):  
        logger.info(f" | removed '{csv_file}' | ")
        os.remove(csv_file)  
        
    csv_files = os.listdir(CSV_TMP)
    for file in csv_files:
        if file.startswith(task_id+"_horcrux"):
            os.remove(os.path.join(CSV_TMP, file))  
        
    custom_speaker_json = os.path.join(CUSTOMSPEAKERPATH, task_id+".json")
    if os.path.isfile(custom_speaker_json):
        with open(custom_speaker_json, 'r') as speakers_file:  
            speakers = json.load(speakers_file)
            for speaker in speakers:
                audio = os.path.join(SPEAKERFOLDER, speaker['audio'])   
                if os.path.isfile(audio):
                    os.remove(audio)  
        logger.info(f" | removed '{custom_speaker_json}' and all custom speakers in task '{task_id}' | ")
        os.remove(custom_speaker_json)  
        
    output_path = os.path.join(OUTPUTPATH, task_id)  
    if not os.path.exists(output_path):  
        logger.error(f" | Task output folder not found. Please check the task ID is correct | ")  
        return BaseResponse(status="FAILED", message=f" | Task output folder not found. Please check the task ID is correct | ", data=False)  
    else:  
        logger.info(f" | removed task output folder '{output_path}' | ")
        shutil.rmtree(output_path)  
      
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
      
    if task_manager.get(task_id):
        generate_state = state_check(task_id, task_manager[task_id]["generate"])
        if isinstance(generate_state, BaseResponse):  
            return generate_state
        quality_state  = state_check(task_id, task_manager[task_id]["quality"])
        if isinstance(quality_state, BaseResponse):  
            return quality_state
        if generate_state or quality_state:
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
        logger.info(f" | ZIP archive {task_id}.zip completed. zip time: {end-start}  | ")  
      
    logger.info(f" | Try to download '{task_id}.zip'. | ")  
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
    
    if task_manager.get(task_id):
        generate_state = state_check(task_id, task_manager[task_id]["generate"])
        if isinstance(generate_state, BaseResponse):  
            return generate_state
        quality_state  = state_check(task_id, task_manager[task_id]["quality"])
        if isinstance(quality_state, BaseResponse):  
            return quality_state
        if generate_state or quality_state:
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
def txt2csv(task_id: str =  None , expansion_ratio: float = 1.0, file: UploadFile = File(...)):  
    # Load the text sample to be generated  
    logger.info(f" | Start to load text sample | ")  
      
    if file.filename.endswith(".txt"):  
        texts = file.file.read().decode('utf-8').splitlines()  
        text_num = len(texts)  
    else:  
        logger.error(f" | We only support txt file | ")  
        return BaseResponse(status="FAILED", message=f" | We only support txt file | ", data=False)  
  
    # Load speakers  
    logger.info(f" | Start to load speakers | ")  
    with open(SPEAKERS, 'r') as speakers_file:  
        speakers = json.load(speakers_file)  
    
    if task_id:
        custom_speaker_json = os.path.join(CUSTOMSPEAKERPATH, task_id+".json")
        if os.path.isfile(custom_speaker_json):
            with open(custom_speaker_json, 'r') as speakers_file:  
                custom_speakers = json.load(speakers_file)  
                for speaker in custom_speakers:
                   speakers.append(speaker)
        
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
        csv_filename = os.path.join(CSV_TMP, f"{task_id}_generated.csv")  
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
    return FileResponse(csv_filename, media_type='application/csv', filename=f"{task_id}.csv")  
    
  
##############################################################################  

@app.get("/speaker_list")
def speaker_list(task_id: str):
    speaker_list = []
    # Load speakers  
    logger.info(f" | Start to load speakers | ")  
    with open(SPEAKERS, 'r') as speakers_file:  
        speakers = json.load(speakers_file)  
    
    if task_id:
        custom_speaker_json = os.path.join(CUSTOMSPEAKERPATH, task_id+".json")
        if os.path.isfile(custom_speaker_json):
            logger.info(f" | Found custom speakers. Start to load custom speakers | ")  
            with open(custom_speaker_json, 'r') as speakers_file:  
                custom_speakers = json.load(speakers_file)  
                for speaker in custom_speakers:
                   speakers.append(speaker)
        
    for speaker in speakers:
        speaker_list.append(speaker['audio'])
    speaker_num = len(speaker_list)
    
    logger.info(f" | Total '{speaker_num}' speakers found | ")
    return BaseResponse(status="OK", message=f" | Total '{speaker_num}' speakers found | ", data=speaker_list) 

@app.post("/start_single_generate_service")
def start_single_generate_service(task_id: str = Form(...)):
    if task_id not in single_generate_state["usage_list"]:
        single_generate_state["usage_list"].append(task_id)

    try:
        sg_state = start_service(task_id, single_generate_state)
        if isinstance(sg_state, BaseResponse):  
            return sg_state  
    except Exception as e:  
        logger.error(f" | An error occurred when start service : {e} | task ID: {task_id} | ")  
        return BaseResponse(status="FAILED", message=f" | An error occurred when start service : {e} | task ID: {task_id} | ", data=False)
        
    logger.info(f" | single generate service already started and task added to usage list | ")
    return BaseResponse(status="OK", message=f" | single generate service already started and add task added to usage list | ", data=True) 
            
@app.post("/single_generate")
async def single_generate(  
    task_id: str = Form(...),  
    speaker_name: str = Form(...),  
    text: str = Form(...),  
):  
    # check service is ready if not start it
    try:
        if task_id not in single_generate_state["usage_list"]:
            single_generate_state["usage_list"].append(task_id)
        sg_state = start_service(task_id, single_generate_state)
        if isinstance(sg_state, BaseResponse):  
            return sg_state  
    except Exception as e:  
        logger.error(f" | An error occurred when start service : {e} | task ID: {task_id} | ")  
        return BaseResponse(status="FAILED", message=f" | An error occurred when start service : {e} | task ID: {task_id} | ", data=False)
    
    # get speaker text from json  
    try:  
        with open(SPEAKERS, 'r') as speakers_file:  
            speakers = json.load(speakers_file)  
    except Exception as e:  
        logger.error(f" | Error reading speakers file: {e} | ")  
        return BaseResponse(status="FAILED", message=f" | Error reading speakers file: {e} | ", data=False)  
        
    if task_id:
        custom_speaker_json = os.path.join(CUSTOMSPEAKERPATH, task_id+".json")
        if os.path.isfile(custom_speaker_json):
            logger.info(f" | Found custom speakers. Start to load custom speakers | ")  
            with open(custom_speaker_json, 'r') as speakers_file:  
                custom_speakers = json.load(speakers_file)  
                for speaker in custom_speakers:
                   speakers.append(speaker)
    
    speaker_text = next((speaker['sentence'] for speaker in speakers if speaker_name == speaker['audio']), "")  
    if speaker_text == "":
        logger.error(f" | Can't find speaker text with provided speaker '{speaker_name}' | ")  
        return BaseResponse(status="FAILED", message=f" | Can't find speaker text with provided speaker '{speaker_name}' | ", data=False) 
    
    # name output file
    now = datetime.now()  
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")  
    output_name = f"{task_id}_{speaker_name[:-4]}_{current_time}.wav"
    
    info = {
        "task_id": task_id,
        "speaker": speaker_name,
        "speaker_text": speaker_text,
        "text": text,
        "output_name": output_name,
    }
    
    if single_generate_state["task_queue"] is not None:
        single_generate_state["task_queue"].put(info)
    else:
        logger.error(f" | something got wrong: single generate service not found | ")
        return BaseResponse(status="FAILED", message=f" | something got wrong: single generate service not found | ", data=False) 
    
    logger.info(f" | task '{task_id}' generate mission received | ")
    return BaseResponse(status="OK", message=f" | task '{task_id}' generate mission received | later use get(/get_single_audio) to get result | ", data=True) 
    
    """
    # get return audio
    for _ in range(5):
        time.sleep(5)
        if single_generate_state["conn"].poll(): 
            single_generate_state["readied_audio"].append(single_generate_state["conn"].recv())

        for ready_file in single_generate_state["readied_audio"]:
            if ready_file.get(task_id):
                audio = ready_file.get(task_id)
                logger.info(f" | Get audio {os.path.basename(audio)} from single inference. | ")
                return FileResponse(audio, media_type='audio/wav', filename=f"{output_name}")  

    logger.info(f" | task '{task_id}: ' time out | ")
    return BaseResponse(status="FAILED", message=f" | task '{task_id}: ' time out | ", data=False) 
    """ 
    
@app.get("/get_single_audio")
def get_single_audio(task_id: str):
    
    try:
        if single_generate_state['audio_queue']:
            while not single_generate_state['audio_queue'].empty():
                audio_info = single_generate_state['audio_queue'].get()
                # get current task id
                current_task_id = next(iter(audio_info))
                
                existing_index = -1  
                # check if task id already in readied audio
                for idx, item in enumerate(single_generate_state["readied_audio"]):  
                    if current_task_id in item:  
                        existing_index = idx  
                        break  
                    
                # if task id already in readied audio, update it
                if existing_index >= 0:  
                    single_generate_state["readied_audio"][existing_index] = audio_info  
                # if task id not in readied audio, append it
                else:  
                    single_generate_state["readied_audio"].append(audio_info)  
        else:
            logger.info(" | single generate service has not been started. please start first | ")
            return BaseResponse(status="FAILED", message=f" | single generate service has not been started. please start first | ", data=False) 
    except BrokenPipeError as e:
        logger.info(" | single generate service has not been started. please start first | ")
        return BaseResponse(status="FAILED", message=f" | single generate service has not been started. please start first | ", data=False) 
    except ConnectionResetError as e:  
        logger.info(" | single generate service has not been started. please start first | ")
        return BaseResponse(status="FAILED", message=f" | single generate service has not been started. please start first | ", data=False) 
    except Exception as e:  
        logger.error(" | An error occurred when get single audio : {e} | task ID: {task_id} | ")
        return BaseResponse(status="FAILED", message=f" | An error occurred when get single audio : {e} | task ID: {task_id} | ", data=False) 

    if task_id not in single_generate_state["usage_list"]:
        logger.info(f" | task '{task_id}' not in usage list please choose a speaker and generate audio first. | ")
        return BaseResponse(status="FAILED", message=f" | task '{task_id}' not in usage list please choose a speaker and generate audio first. | task ID: {task_id} | ", data=False) 
    
    print(single_generate_state["readied_audio"])        
    for ready_file in single_generate_state["readied_audio"]:
        if ready_file.get(task_id):
            audio = ready_file.get(task_id)
            logger.info(f" | Get audio {os.path.basename(audio)} from single inference. | ")
            return FileResponse(audio, media_type='audio/wav', filename=f"{os.path.basename(audio)}")  
    
    logger.info(f" | task '{task_id}' haven't finish generate please try again later | ")
    return BaseResponse(status="OK", message=f" | task '{task_id}' haven't finish generate please try again later | ", data=True) 


@app.post("/stop_single_generate_service")
def stop_single_generate_service(
    task_id: str = Form(...),
    ):
    if task_id in single_generate_state["usage_list"]:
        single_generate_state["usage_list"].remove(task_id)
        logger.info(f" | Got task '{task_id}' | removed from usage list | ")
    else:
        logger.info(f" | task '{task_id}' not in usage list | skip end process | ")
        return BaseResponse(status="OK", message=f" | task '{task_id}' not in usage list | skip end process | ", data=True) 
    
    if single_generate_state["usage_list"] == []:
        logger.info(f" | No task in usage list. Start to end the single generate service | ")
        stop_info = stop_task(task_id, [single_generate_state["conn"]], [None])
        if stop_info:
            return stop_info  
        else:
            logger.info(f" | single generate service has been closed | ")
            return BaseResponse(status="OK", message=f" | Got task '{task_id}' | removed from usage list and no task in usage closed single generate service | ", data=True) 
    
    return BaseResponse(status="OK", message=f" | Got task '{task_id}' | removed from usage list | ", data=True) 

##############################################################################  

@app.get("/thread_suggestion")
def thread_suggestion(
    ):
    try:  
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],  
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)  
          
        memory_free = [int(x) for x in result.stdout.strip().split('\n')]  
        memory = sum(memory_free)
        available_thread = int(memory // 16 // 1024)
        available_thread = available_thread if available_thread <= 3 else 3
        if available_thread <= 0:
            logger.error(" | No available GPU memory for processing. | ")
            return BaseResponse(status="FAILED", message=f" | No available GPU memory. | ", data=None)
        logger.info(f" | Available GPU memory: {memory} MB | Suggested threads: {available_thread} | ")
        return BaseResponse(status="OK", message=f" | Available GPU memory: {memory} MB | Suggested threads: {available_thread} | ", data=available_thread)
    except subprocess.CalledProcessError as e:  
        logger.error("An error occurred while trying to query GPU memory:", e)  
        return BaseResponse(status="FAILED", message=f" | An error occurred while trying to query GPU memory: {e} | ", data=None)

@app.get("/show_logs")
def show_logs():
    """   
    Show the latest logs from the log file.  
      
    :return: BaseResponse  
        A response object containing the latest logs.  
    """  
    log_file_path = os.path.join(LOGPATH, "generate.log")  
    if not os.path.exists(log_file_path):  
        logger.error(f" | Log file not found: {log_file_path} | ")  
        return BaseResponse(status="FAILED", message=f" | Log file not found: {log_file_path} | ", data=None)  
    
    try:  
        with open(log_file_path, 'r') as log_file:  
            logs = log_file.readlines()[-100:]  
            LOG_PATTERN = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - generate_logger - INFO - (.*)')  
            cleaned_logs = []  
            for log in logs:  
                match = LOG_PATTERN.match(log)  
                if match:  
                    timestamp, message = match.groups()  
                    cleaned_logs.append(f"{timestamp} - {message.strip()}")  
                else:  
                    logger.debug(f"No match for line: {log.strip()}")   
        return BaseResponse(status="OK", message=" | Latest logs retrieved successfully | ", data=cleaned_logs)  
    except Exception as e:  
        logger.error(f" | Error reading log file: {e} | ")  
        return BaseResponse(status="FAILED", message=f" | Error reading log file: {e} | ", data=None)
    
@app.get("/speaker_audition")
def speaker_audition(task_id: str, speaker_name: str):
    """   
    Get the audio file for a specific speaker in a task.  
      
    :param task_id: str  
        The ID of the task to get the speaker audio for.  
    :param speaker_name: str  
        The name of the speaker whose audio file is requested.  
    :return: BaseResponse or FileResponse  
        The audio file if found, otherwise an error response.  
    """  
    audio_file = os.path.join(SPEAKERFOLDER, speaker_name)  
    
    if not os.path.isfile(audio_file):  
        logger.error(f" | Audio file for speaker '{speaker_name}' not found in task '{task_id}' | ")  
        return BaseResponse(status="FAILED", message=f" | Audio file for speaker '{speaker_name}' not found in task '{task_id}' | ", data=False)  
    
    logger.info(f" | Found audio file for speaker '{speaker_name}' in task '{task_id}' | ")  
    return FileResponse(audio_file, media_type='audio/wav', filename=os.path.basename(audio_file))

@app.get("/check_single_generate_service_permission")
def check_single_generate_service_permission(task_id: str):
    if task_id not in single_generate_state["usage_list"]:
        logger.info(f" | task '{task_id}' not in usage list | ")
        return BaseResponse(status="OK", message=f" | task '{task_id}' not in usage list | ", data=False)
    else:
        logger.info(f" | task '{task_id}' in usage list | ")
        return BaseResponse(status="OK", message=f" | task '{task_id}' in usage list | ", data=True)

@app.post("/cleanup_inactive_tasks")
def cleanup_inactive_tasks():
    """
    Manually trigger cleanup of inactive tasks in usage_list.
    """
    try:
        initial_count = len(single_generate_state["usage_list"])
        check_and_cleanup_inactive_tasks()
        final_count = len(single_generate_state["usage_list"])
        removed_count = initial_count - final_count
        
        return BaseResponse(
            status="OK", 
            message=f" | Cleanup completed. Removed {removed_count} inactive tasks | ", 
            data={
                "initial_count": initial_count,
                "final_count": final_count,
                "removed_count": removed_count,
                "current_usage_list": single_generate_state["usage_list"]
            }
        )
    except Exception as e:
        logger.error(f" | Error during manual cleanup: {e} | ")
        return BaseResponse(status="FAILED", message=f" | Error during manual cleanup: {e} | ", data=False)

    
##############################################################################  

def check_and_cleanup_inactive_tasks():
    """
    Check for inactive tasks in usage_list and remove them if their readied_audio 
    state hasn't changed for more than 1 hour.
    """
    current_time = datetime.now()
    tasks_to_remove = []
    
    # Get current state of readied_audio for each task
    current_state = {}
    for ready_file in single_generate_state["readied_audio"]:
        for task_id in ready_file.keys():
            if task_id in single_generate_state["usage_list"]:
                current_state[task_id] = ready_file[task_id]
    
    # Check for tasks in usage_list that don't have readied_audio
    for task_id in single_generate_state["usage_list"]:
        if task_id not in current_state:
            current_state[task_id] = None
    
    # Check each task in usage_list
    for task_id in single_generate_state["usage_list"]:
        if task_id not in usage_list_history:
            # First time seeing this task, record its state
            usage_list_history[task_id] = {
                "last_check_time": current_time,
                "last_audio_state": current_state.get(task_id),
                "state_unchanged_since": current_time
            }
        else:
            # Task exists in history, check if state changed
            last_audio_state = usage_list_history[task_id]["last_audio_state"]
            current_audio_state = current_state.get(task_id)
            
            if last_audio_state == current_audio_state:
                # State hasn't changed, check if it's been more than 1 hour
                time_since_last_change = current_time - usage_list_history[task_id]["state_unchanged_since"]
                if time_since_last_change.total_seconds() >= 3600:  # 1 hour = 3600 seconds
                    logger.info(f" | Task '{task_id}' has been inactive for {time_since_last_change}, marking for removal | ")
                    tasks_to_remove.append(task_id)
            else:
                # State changed, update the history
                usage_list_history[task_id]["last_audio_state"] = current_audio_state
                usage_list_history[task_id]["state_unchanged_since"] = current_time
            
            usage_list_history[task_id]["last_check_time"] = current_time
    
    # Remove inactive tasks
    for task_id in tasks_to_remove:
        try:
            if task_id in single_generate_state["usage_list"]:
                single_generate_state["usage_list"].remove(task_id)
                logger.info(f" | Removed inactive task '{task_id}' from usage list | ")
            
            # Clean up history for removed task
            if task_id in usage_list_history:
                del usage_list_history[task_id]
                
            # Clean up readied_audio for removed task
            single_generate_state["readied_audio"] = [
                ready_file for ready_file in single_generate_state["readied_audio"] 
                if task_id not in ready_file
            ]
            
        except Exception as e:
            logger.error(f" | Error removing inactive task '{task_id}': {e} | ")
    
    # Clean up history for tasks no longer in usage_list
    tasks_in_history = list(usage_list_history.keys())
    for task_id in tasks_in_history:
        if task_id not in single_generate_state["usage_list"]:
            del usage_list_history[task_id]
    
    # If no tasks remain in usage_list, stop the single generate service
    if single_generate_state["usage_list"] == [] and single_generate_state["conn"] is not None:
        logger.info(" | No tasks remaining in usage list, stopping single generate service | ")
        try:
            stop_info = stop_task("auto_cleanup", [single_generate_state["conn"]], [None])
            if not stop_info:
                logger.info(" | Single generate service stopped automatically | ")
        except Exception as e:
            logger.error(f" | Error stopping single generate service: {e} | ")

# Clean up audio files  
def delete_old_audio_files():  
    """   
    The process of deleting old audio files  
      
    :param None: The function does not take any parameters  
    :rtype: None: The function does not return any value  
    :logs: Deleted old files  
    """  
    current_time = time.time()  
    audio_dirs = ["./audio", os.path.join(OUTPUTPATH,"tmp")]
    for audio_dir in audio_dirs:
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
                    
    for csv_file in os.listdir(CSV_TMP):
        if csv_file.endswith("_generated.csv"):
            file_path = os.path.join(CSV_TMP, csv_file)
            if os.path.isfile(file_path):
                file_creation_time = os.path.getctime(file_path)
                os.remove(file_path)
                logger.info(f"Deleted old CSV file: {file_path}")
    
  
# Daily task scheduling  
def schedule_daily_task(stop_event, local_now):  
    """   
    Schedule a daily cleanup task that removes outdated audio files at midnight,
    and hourly cleanup of inactive tasks from usage_list.
      
    :param stop_event: threading.Event  
        A signal used to stop the scheduled task gracefully.  
    """  
    last_hour_check = local_now.hour
    
    while not stop_event.is_set():  
        current_time = datetime.now(pytz.timezone('Asia/Taipei'))
        
        # Daily cleanup at midnight
        if current_time.hour == 0 and current_time.minute == 0:  
            delete_old_audio_files()  
            time.sleep(60)  # Prevent triggering multiple times within the same minute  
        
        # Hourly cleanup of inactive tasks
        if current_time.hour != last_hour_check:
            logger.info(f" | Starting hourly cleanup of inactive tasks at {current_time.strftime('%Y-%m-%d %H:%M:%S')} | ")
            check_and_cleanup_inactive_tasks()
            last_hour_check = current_time.hour
            
        time.sleep(60)  # Check every minute  
        
###############################################################################
###################### This is for OpenAI compatible TTS ######################
###############################################################################

@app.post("/v1/audio/speech")
async def openai_compatible_tts(request: OpenAITTSRequest):
    """
    OpenAI compatible TTS endpoint for Open Notebook integration.
    
    Compatible with OpenAI's /v1/audio/speech API format.
    """
    try:
        # 使用固定的 task_id
        task_id = "openai_NB_TTS"
        
        # 記錄請求
        logger.info(f" | OpenAI TTS request | task_id: {task_id} | voice: {request.voice} | text length: {len(request.input)} | ")
        
        # 確保 service 啟動
        if task_id not in single_generate_state["usage_list"]:
            single_generate_state["usage_list"].append(task_id)
        
        sg_state = start_service(task_id, single_generate_state)
        if isinstance(sg_state, BaseResponse):
            raise HTTPException(status_code=500, detail=sg_state.message)
        
        # 加載 speakers
        try:
            with open(SPEAKERS, 'r') as speakers_file:
                speakers = json.load(speakers_file)
        except Exception as e:
            logger.error(f" | Error reading speakers file: {e} | ")
            raise HTTPException(status_code=500, detail=f"Error loading speakers: {e}")
        
        # 加載所有 custom speakers (遍歷 CUSTOMSPEAKERPATH 目錄下的所有 JSON 文件)
        if os.path.isdir(CUSTOMSPEAKERPATH):
            for custom_file in os.listdir(CUSTOMSPEAKERPATH):
                if custom_file.endswith('.json'):
                    custom_speaker_json = os.path.join(CUSTOMSPEAKERPATH, custom_file)
                    try:
                        with open(custom_speaker_json, 'r') as speakers_file:
                            custom_speakers = json.load(speakers_file)
                            speakers.extend(custom_speakers)
                    except Exception as e:
                        logger.warning(f" | Error loading custom speaker file {custom_file}: {e} | ")
        
        # 查找 speaker text
        speaker_text = next(
            (speaker['sentence'] for speaker in speakers if request.voice == speaker['audio']),
            None
        )
        
        if not speaker_text:
            # 如果沒找到完整匹配,嘗試模糊匹配 (去掉 .wav 後綴)
            voice_name = request.voice.replace('.wav', '')
            speaker_text = next(
                (speaker['sentence'] for speaker in speakers 
                 if voice_name in speaker['audio'] or speaker['audio'].replace('.wav', '') == voice_name),
                None
            )
        
        if not speaker_text:
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{request.voice}' not found. Available voices: {[s['audio'] for s in speakers[:5]]}"
            )
        
        # 生成輸出文件名
        now = datetime.now()
        current_time = now.strftime("%Y%m%d_%H%M%S")
        output_name = f"{task_id}_{current_time}.wav"
        
        # 構建生成任務
        info = {
            "task_id": task_id,
            "speaker": request.voice if request.voice.endswith('.wav') else f"{request.voice}.wav",
            "speaker_text": speaker_text,
            "text": request.input,
            "output_name": output_name,
        }
        
        # 提交任務到隊列
        if single_generate_state["task_queue"] is None:
            raise HTTPException(status_code=500, detail="TTS service not ready")
        
        single_generate_state["task_queue"].put(info)
        logger.info(f" | OpenAI TTS task queued | task_id: {task_id} | ")
        
        # 等待音頻生成完成 (同步等待,最多 60 秒)
        max_wait_time = 60  # 秒
        check_interval = 0.5  # 秒
        waited = 0
        
        while waited < max_wait_time:
            time.sleep(check_interval)
            waited += check_interval
            
            # 檢查是否有新的音頻完成 (從 audio_queue 獲取)
            logger.debug(f" | Checking queue, waited: {waited}s | ")
            if single_generate_state["audio_queue"] and not single_generate_state["audio_queue"].empty():
                try:
                    audio_info = single_generate_state["audio_queue"].get_nowait()
                    # audio_info 已經是 dict 格式: {task_id: audio_path}
                    single_generate_state["readied_audio"].append(audio_info)
                    logger.info(f" | Received audio from queue: {audio_info} | ")
                except Exception as e:
                    logger.error(f" | Queue get error: {e} | ")
            
            # 檢查是否是我們要的音頻 (用 output_name 精確匹配)
            for ready_file in single_generate_state["readied_audio"]:
                if task_id in ready_file:
                    audio_path = ready_file[task_id]
                    # 用 output_name 精確匹配，確保是這次請求生成的音檔
                    if audio_path.endswith(output_name):
                        logger.info(f" | OpenAI TTS completed | audio: {os.path.basename(audio_path)} | ")
                        
                        # 從 readied_audio 中移除已返回的音檔，避免重複返回
                        single_generate_state["readied_audio"].remove(ready_file)
                        
                        # 返回音頻文件
                        return FileResponse(
                            audio_path,
                            media_type='audio/wav',
                            filename=output_name,
                            headers={
                                "Content-Disposition": f"attachment; filename={output_name}"
                            }
                        )
        
        # 超時
        logger.warning(f" | OpenAI TTS timeout | task_id: {task_id} | ")
        raise HTTPException(status_code=504, detail="TTS generation timeout")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" | OpenAI TTS error: {e} | ")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_openai_models():
    """
    List available TTS models (OpenAI compatible).
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "breezyvoice-zh",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "breezyvoice",
                "permission": [],
                "root": "breezyvoice-zh",
                "parent": None,
            }
        ]
    }


@app.get("/v1/voices")
async def list_available_voices():
    """
    List all available voices (speakers).
    Useful for Open Notebook to discover available voices.
    """
    try:
        with open(SPEAKERS, 'r') as speakers_file:
            speakers = json.load(speakers_file)
        
        voices = [
            {
                "voice_id": speaker['audio'],
                "name": speaker['audio'].replace('.wav', ''),
                "preview_url": None,
                "description": speaker.get('sentence', '')[:50] + "..."
            }
            for speaker in speakers
        ]
        
        return {
            "voices": voices,
            "count": len(voices)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
###############################################################################
########################### This is for Bobo Agent ############################
###############################################################################

@app.post("/bobo/single_generate")
async def bobo_single_generate(
    task_id: str = Form(...),
    speaker_name: str = Form(...),
    text: str = Form(...),
    timeout: int = Form(60),  # 可選的超時時間，默認 60 秒
):
    """
    單次生成音頻並等待結果返回。
    專門給 Bobo Agent 服務使用，發送請求後會同步等待音頻生成完成才返回。
    
    :param task_id: 任務 ID
    :param speaker_name: 語者名稱 (例如: jack10.wav)
    :param text: 要合成的文字
    :param timeout: 超時時間 (秒)，默認 60 秒
    :return: 生成的音頻文件
    """
    try:
        # 確保 service 啟動
        if task_id not in single_generate_state["usage_list"]:
            single_generate_state["usage_list"].append(task_id)
        
        sg_state = start_service(task_id, single_generate_state)
        if isinstance(sg_state, BaseResponse):
            return sg_state
        
        # 加載 speakers
        try:
            with open(SPEAKERS, 'r') as speakers_file:
                speakers = json.load(speakers_file)
        except Exception as e:
            logger.error(f" | Error reading speakers file: {e} | ")
            return BaseResponse(status="FAILED", message=f" | Error reading speakers file: {e} | ", data=False)
        
        # 加載所有 custom speakers
        if os.path.isdir(CUSTOMSPEAKERPATH):
            for custom_file in os.listdir(CUSTOMSPEAKERPATH):
                if custom_file.endswith('.json'):
                    custom_speaker_json = os.path.join(CUSTOMSPEAKERPATH, custom_file)
                    try:
                        with open(custom_speaker_json, 'r') as speakers_file:
                            custom_speakers = json.load(speakers_file)
                            speakers.extend(custom_speakers)
                    except Exception as e:
                        logger.warning(f" | Error loading custom speaker file {custom_file}: {e} | ")
        
        # 查找 speaker text
        speaker_text = next(
            (speaker['sentence'] for speaker in speakers if speaker_name == speaker['audio']),
            None
        )
        
        if not speaker_text:
            # 嘗試模糊匹配
            voice_name = speaker_name.replace('.wav', '')
            speaker_text = next(
                (speaker['sentence'] for speaker in speakers 
                 if voice_name in speaker['audio'] or speaker['audio'].replace('.wav', '') == voice_name),
                None
            )
        
        if not speaker_text:
            logger.error(f" | Can't find speaker text with provided speaker '{speaker_name}' | ")
            return BaseResponse(status="FAILED", message=f" | Can't find speaker text with provided speaker '{speaker_name}' | ", data=False)
        
        # 生成輸出文件名
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        speaker_base = speaker_name[:-4] if speaker_name.endswith('.wav') else speaker_name
        output_name = f"{task_id}_{speaker_base}_{current_time}.wav"
        
        # 構建生成任務
        info = {
            "task_id": task_id,
            "speaker": speaker_name if speaker_name.endswith('.wav') else f"{speaker_name}.wav",
            "speaker_text": speaker_text,
            "text": text,
            "output_name": output_name,
        }
        
        # 提交任務到隊列
        if single_generate_state["task_queue"] is None:
            logger.error(f" | single generate service not found | ")
            return BaseResponse(status="FAILED", message=f" | single generate service not found | ", data=False)
        
        single_generate_state["task_queue"].put(info)
        logger.info(f" | Bobo Agent task '{task_id}' queued | speaker: {speaker_name} | ")
        
        # 等待音頻生成完成
        check_interval = 0.5  # 秒
        waited = 0
        
        while waited < timeout:
            time.sleep(check_interval)
            waited += check_interval
            
            # 從 audio_queue 獲取完成的音頻
            if single_generate_state["audio_queue"] and not single_generate_state["audio_queue"].empty():
                try:
                    audio_info = single_generate_state["audio_queue"].get_nowait()
                    current_task_id = next(iter(audio_info))
                    
                    # 直接添加到 readied_audio，不覆蓋舊的
                    single_generate_state["readied_audio"].append(audio_info)
                    
                    logger.info(f" | Received audio from queue: {audio_info} | ")
                except Exception as e:
                    logger.error(f" | Queue get error: {e} | ")
            
            # 檢查是否是我們要的音頻 (用 output_name 精確匹配)
            for ready_file in single_generate_state["readied_audio"]:
                if task_id in ready_file:
                    audio_path = ready_file[task_id]
                    # 用 output_name 精確匹配，確保是這次請求生成的音檔
                    if audio_path.endswith(output_name):
                        logger.info(f" | Bobo Agent task completed | audio: {os.path.basename(audio_path)} | ")
                        
                        # 從 readied_audio 中移除已返回的音檔
                        single_generate_state["readied_audio"].remove(ready_file)
                        
                        # 返回音頻文件
                        return FileResponse(
                            audio_path,
                            media_type='audio/wav',
                            filename=output_name
                        )
        
        # 超時
        logger.warning(f" | Bobo Agent task '{task_id}' timeout after {timeout}s | ")
        return BaseResponse(status="FAILED", message=f" | task '{task_id}' timeout after {timeout}s | ", data=False)
        
    except Exception as e:
        logger.error(f" | Bobo Agent error: {e} | task_id: {task_id} | ")
        return BaseResponse(status="FAILED", message=f" | An error occurred: {e} | ", data=False)

    
##############################################################################

    
# Signal handler for graceful shutdown  
def handle_exit(sig, frame):  
    """   
    Handle system exit signals (SIGINT, SIGTERM) to gracefully shut down the daily task scheduler.  
    """  
    logger.info("Received shutdown signal, cleaning up...")
    stop_event.set()  
    task_thread.join(timeout=5)  
    
    # 終止子進程
    if single_generate_state["process"] is not None:
        try:
            single_generate_state["process"].terminate()
            single_generate_state["process"].join(timeout=3)
            if single_generate_state["process"].is_alive():
                single_generate_state["process"].kill()
                logger.info("Force killed single generate process.")
        except Exception as e:
            logger.error(f"Error terminating process: {e}")
    
    # 關閉連接
    if single_generate_state["conn"] is not None:
        try:
            single_generate_state["conn"].close()
        except Exception:
            pass
            
    logger.info("Scheduled task has been stopped.")  
    os._exit(0)  

if __name__ == "__main__":  
    multiprocessing.set_start_method('spawn') 
    logger, tz, local_now = initialize_logging_and_directories()

    # Start daily task scheduling  
    stop_event = Event()  
    task_thread = Thread(target=schedule_daily_task, args=(stop_event, local_now))  
    task_thread.start()  
    delete_old_audio_files()  
    signal.signal(signal.SIGINT, handle_exit)  
    signal.signal(signal.SIGTERM, handle_exit)  
    
    port = int(os.environ.get("PORT", 80))  
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)  

    
     