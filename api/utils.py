import os  
import csv 
import math  
import time 
import jiwer  
import zipfile  
import pandas as pd  
from typing import Dict, Set  

from lib.constant import Common, CSV_TMP
from lib.log_config import setup_sys_logging
from lib.base_object import BaseResponse  


logger = setup_sys_logging()  
  
def calculate_cer(reference: str, hypothesis: str) -> float:  
    """  
    Calculate the Character Error Rate (CER) between two strings.  
      
    :param reference: str  
        The reference string (ground truth).  
    :param hypothesis: str  
        The hypothesis string (predicted transcription).  
    :return: float  
        The CER value.  
    """  
    return jiwer.cer(reference, hypothesis)  
  
def load_csv_data(csv_file_path: str) -> Dict[str, str]:  
    """  
    Load data from a CSV file into a dictionary.  
      
    :param csv_file_path: str  
        The path to the CSV file.  
    :return: dict  
        A dictionary where the keys are the output audio filenames and the values are the content to synthesize.  
    """  
    data = {}  
    with open(csv_file_path, mode='r', encoding='utf-8') as file:  
        csv_reader = csv.DictReader(file)  
        for row in csv_reader:  
            data[row['output_audio_filename']] = row['content_to_synthesize']  
    return data  
  
def load_file_list(file_path: str) -> Set[str]:  
    """  
    Load a list of file names from a text file into a set.  
      
    :param file_path: str  
        The path to the text file.  
    :return: set  
        A set of file names.  
    """  
    if os.path.exists(file_path):  
        with open(file_path, 'r', encoding='utf-8') as file:  
            return set(file.read().splitlines())  
    return set()  
  
def save_file_list(file_path: str, file_list: Set[str]) -> None:  
    """  
    Save a list of file names to a text file.  
      
    :param file_path: str  
        The path to the text file.  
    :param file_list: set  
        A set of file names to be saved.  
    :rtype: None  
    """  
    with open(file_path, 'w', encoding='utf-8') as file:  
        for item in file_list:  
            file.write(f"{item}\n")  
  
def zip_wav_files(file_folder: str) -> None:  
    """  
    Compress all .wav files in a folder into a ZIP archive.  
      
    :param file_folder: str  
        The path to the folder containing .wav files.  
    :rtype: None  
    """  
    output_zip = os.path.join(file_folder, file_folder.split('/')[-1] + '.zip')  
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:  
        for root, dirs, files in os.walk(file_folder):  
            for file in files:  
                if file.lower().endswith('.wav'):  
                    file_path = os.path.join(root, file)  
                    zipf.write(file_path, os.path.relpath(file_path, file_folder))  
                    
                    
def state_check(task_id, processes):
    states = []
    for process in processes:
        if process:
            try:
                process.send(Common.STATE.value)
                states.append(process.recv())
            except BrokenPipeError as e:
                states.append(None)
            except ConnectionResetError as e:  
                states.append(None)
            except Exception as e:
                logger.error(f" | An error occurred when getting task '{task_id}' state: {str(e)} | ")  
                return BaseResponse(status="FAILED", message=f" | An error occurred when getting task '{task_id}' state: {str(e)} | ", data=False)
        else:
            states.append(None)
            
    # Determine the status
    if True in states:
        state = True
    elif False in states:
        state = False
    else:
        state = None

    return state

def send_stop_flag(processes):
    for process in processes:
        if process:
            try:
                process.send(Common.STOP.value)
            except BrokenPipeError as e:
                pass
            except ConnectionResetError as e:  
                pass
                    
def stop_task(task_id, generate_process, quality_process):  
    """   
    Stop the audio generation and model tasks for a given task ID.  
      
    :param task_id: str  
        The ID of the task to stop.  
    :return: BaseResponse  
        A response object indicating the success or failure of the stop operation.  
    """  
    try:
        send_stop_flag(generate_process)
        send_stop_flag(quality_process)
    except Exception as e:  
        logger.error(f" | An error occurred when cancel task : {e} | task ID: {task_id} | ")  
        return BaseResponse(status="FAILED", message=f" | An error occurred when cancel task : {e} | task ID: {task_id} | ", data=False)
    
    for _ in range(10):  
        generate_state = state_check(task_id, generate_process)
        quality_state = state_check(task_id, quality_process)
        if generate_state is None and quality_state is None:  
            logger.info(f" | Task '{task_id}' all process has been stopped. | ")  
            return  
        time.sleep(5)  
        send_stop_flag(generate_process)
        send_stop_flag(quality_process)
        logger.info(f" | Waiting for task '{task_id}' to stop... | ")  
        
    logger.error(f" | Try to stop Task '{task_id}' has already 10 times. But nothing happens. | ")  
    return BaseResponse(status="FAILED", message=f" | Failed | Try to stop Task '{task_id}' has already 10 times. But nothing happens. | ", data=False)  

def split_csv(csv_file_path: str, num_parts: int):  
    horcruxes = []
    
    # count lines
    df = pd.read_csv(csv_file_path)  
    total_rows = len(df)  
    part_rows = math.ceil(total_rows / num_parts)  
    
    # split and save to horcrux of csv
    for i in range(num_parts):  
        start_row = i * part_rows  
        end_row = start_row + part_rows  
        part_df = df[start_row:end_row]  
        
        csv_name = os.path.basename(csv_file_path)
        horcrux_name = f"{csv_name[:-4]}_horcrux{i + 1}.csv"  
        horcrux_path = os.path.join(CSV_TMP, horcrux_name)  
        part_df.to_csv(horcrux_path, index=False)   
        horcruxes.append(horcrux_path)
  
    return horcruxes  

                    
    