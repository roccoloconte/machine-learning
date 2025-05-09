import os
import logging
import time
from datetime import datetime

def setup_logging(log_dir='./image/logs'):
    """Set up logging configuration
    
    Args:
        log_dir: Directory to save log files
        
    Returns:
        log_file: Path to the log file
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger()
    logger.info(f"Logging to {log_file}")
    
    return log_file 