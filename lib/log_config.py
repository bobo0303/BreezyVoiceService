import logging  
from logging.handlers import RotatingFileHandler  
import datetime  
import pytz  
import os  
  
class TaipeiFormatter(logging.Formatter):  
    def converter(self, timestamp):  
        # 將 UTC 時間轉換為台北時間  
        dt = datetime.datetime.fromtimestamp(timestamp, pytz.utc)  
        return dt.astimezone(pytz.timezone('Asia/Taipei'))  
  
    def formatTime(self, record, datefmt=None):  
        dt = self.converter(record.created)  
        if datefmt:  
            return dt.strftime(datefmt)  
        else:  
            # 如果沒有提供 datefmt，則使用默認格式  
            return dt.strftime("%Y-%m-%d %H:%M:%S")  
  
def setup_logger(logger_name, log_file):  
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  
      
    logger = logging.getLogger(logger_name)  
      
    if not logger.handlers:  
        logger.setLevel(logging.INFO)  
          
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)  
        file_handler.setFormatter(TaipeiFormatter(log_format))  
          
        logger.addHandler(file_handler)  
          
    return logger  
  
def setup_sys_logging():  
    return setup_logger('sys_logger', 'logs/sys.log')  
  
def setup_generate_logging():  
    return setup_logger('generate_logger', 'logs/generate.log')  
  
def setup_whisper_logging():  
    return setup_logger('whisper_logger', 'logs/whisper.log')  
  
def configure_utc8_time():  
    utc_now = datetime.datetime.now(pytz.utc)  
    tz = pytz.timezone('Asia/Taipei')  
    local_now = utc_now.astimezone(tz)  
    return local_now  
  
# 設置根記錄器，確保所有日誌輸出格式一致  
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  
logging.basicConfig(level=logging.INFO, format=log_format, handlers=[  
    logging.StreamHandler()  # 只保留控制台輸出  
])  