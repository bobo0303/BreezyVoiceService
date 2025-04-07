import os
import json
from lib.constant import TASKPAIR, OUTPUTPATH

def load_task_pairs():  
    if os.path.exists(TASKPAIR):  
        with open(TASKPAIR, 'r') as f:  
            task_pairs = json.load(f)  
        return task_pairs  
    return {}  
  
def initialize_folder_to_task_id():  
    folder_to_task_id = {}
    task_pairs = load_task_pairs()  
    updated_task_pairs = task_pairs.copy()  

    for task_id, output_audio_folder in task_pairs.items():  
        output_path = os.path.join(OUTPUTPATH, output_audio_folder)  
        if os.path.exists(output_path):  
            folder_to_task_id[output_audio_folder] = task_id  
        else:  
            del updated_task_pairs[task_id]  
      
    if not os.path.exists(TASKPAIR.split("/")[0]):  
        os.makedirs(TASKPAIR.split("/")[0])  
      
    # Write the updated task_pairs back to the file  
    with open(TASKPAIR, 'w') as f:  
        json.dump(updated_task_pairs, f, indent=4)  
      
    return folder_to_task_id 

def write_task_pair(task_id, output_audio_folder):
    # Load existing TASKPAIR data  
    if os.path.exists(TASKPAIR):  
        with open(TASKPAIR, 'r') as f:  
            task_pairs = json.load(f)  
    else:  
        task_pairs = {}  

    # Add new task pair  
    task_pairs[task_id] = output_audio_folder  

    # Save updated TASKPAIR data  
    with open(TASKPAIR, 'w') as f:  
        json.dump(task_pairs, f, indent=4)  
  