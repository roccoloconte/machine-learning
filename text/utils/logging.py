import os
import time
import logging
import logging.handlers

def setup_logging():
    """
    Configure logging to write to both a rotating file and the console.
    Returns the path to the log file.
    """
    log_dir = os.path.join("text", "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{time.strftime('%Y%m%d%H%M%S')}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.addHandler(console_handler)
    
    return log_file
