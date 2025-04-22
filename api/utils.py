import os  
import csv  
import jiwer  
import zipfile  
from typing import Dict, Set  
  
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
                    

                    
                    