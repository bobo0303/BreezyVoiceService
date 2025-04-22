from fastapi import FastAPI, UploadFile, File, Form  
from fastapi.responses import FileResponse  
import io
import os  
import csv  
import json
import pytz  
import time  
import random
import shutil  
import signal  
import zipfile  
import uvicorn  
import multiprocessing
from threading import Thread, Event  
from datetime import datetime  
  
from api.threading_api import process_batch_task, quality_checking_task  
from api.utils import load_file_list, zip_wav_files, state_check, stop_task, split_csv

from lib.constant import OUTPUTPATH, CSV_TMP, CSV_HEADER_FORMAT, SPEAKERFOLDER, CUSTOMSPEAKERPATH, SPEAKERS, QUALITY_PASS_TXT, QUALITY_FAIL_TXT
from lib.base_object import BaseResponse  
from lib.log_config import setup_sys_logging  


#############################################################################  
  
# Initialize FastAPI app and model  
app = FastAPI()  
task_manager = {}
  
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
  
@app.get("/speaker_list")
def speaker_list(task_id: str):
    return

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
        "unprocessed": unprocessed_num  
    }  
      
    return BaseResponse(status="OK", message=f" | Task {task_id} is {state}. | progress: {progress}% | not audio: {audio_count} | keep: {keep_num} | del: {delete_num} | unprocessed: {unprocessed_num} | ", data=return_info)  
  
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
    num_thread: int = 3,  
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
    
    csv_horcruxes = split_csv(csv_file_path, num_thread)
      
    output_path = os.path.join(OUTPUTPATH, task_id)  
      
    if os.path.exists(output_path):  
        logger.info(f" | task ID: {task_id} already exists. Keep the old audio and continue generate | ")  
    else:  
        os.makedirs(output_path)  
        logger.info(f" | Start new task | task ID: {task_id} |")  
        
    generate_parent_conns = []
    for horcrux in csv_horcruxes:
        generate_parent_conn, generate_child_conn = multiprocessing.Pipe()
        generate_process = multiprocessing.Process(target=process_batch_task, args=(horcrux, output_path, task_id, quality_check, generate_child_conn))  
        generate_process.start()  
        generate_parent_conns.append(generate_parent_conn)
    
    quality_parent_conns = []
    if quality_check:  
        quality_parent_conn, quality_child_conn = multiprocessing.Pipe()
        quality_process = multiprocessing.Process(target=quality_checking_task, args=(task_id, quality_child_conn))  
        quality_process.start()  
    else:  
        quality_parent_conn = None
        passed_list = os.path.join(output_path, QUALITY_PASS_TXT)  
        failed_list = os.path.join(output_path, QUALITY_FAIL_TXT)  
        if os.path.exists(passed_list):  
            os.remove(passed_list)  
        if os.path.exists(failed_list):  
            os.remove(failed_list)  
            
    quality_parent_conns.append(quality_parent_conn)
    
    # add task into task manager 
    task_manager[task_id] = {"generate": generate_parent_conns, "quality": quality_parent_conns}
    
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
        logger.info(f" | ZIP archive {task_id}.zip completed. zip time: {end-start} | ")  
      
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
                   speakers.append(speakers)
        
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
        csv_filename = os.path.join(CSV_TMP, f"{task_id}.csv")  
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
def schedule_daily_task(stop_event, local_now):  
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
  

  
# Signal handler for graceful shutdown  
def handle_exit(sig, frame):  
    """   
    Handle system exit signals (SIGINT, SIGTERM) to gracefully shut down the daily task scheduler.  
    """  
    stop_event.set()  
    task_thread.join()  
    logger.info("Scheduled task has been stopped.")  
    os._exit(0)  

  
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
    
    
     