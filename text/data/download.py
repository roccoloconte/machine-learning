import requests
import logging
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

DATASETS = {
    'tinyshakespeare': {
        'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'description': 'A tiny subset of Shakespeare\'s work from Andrej Karpathy\'s char-rnn',
        'filename': 'tinyshakespeare.txt'
    }
}

def download_file(url: str, output_path: Path) -> None:
    """
    Download a file from a URL to the specified path.
    
    Args:
        url: URL to download from
        output_path: Path where the file should be saved
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get total file size from content-length header
        total_size = int(response.headers.get('content-length', 0))
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress tracking
        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for data in response.iter_content(chunk_size=8192):
                    if data:  # Filter out keep-alive chunks
                        downloaded += len(data)
                        f.write(data)
                        done = int(25 * downloaded / total_size)
                        print(f"\rDownloading: [{'=' * done}{' ' * (25-done)}] {downloaded:,}/{total_size:,} bytes", 
                              end='', flush=True)
                print("\n", end='')
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading file: {e}")
        raise

def setup_dataset(dataset_name: str, data_dir: Path) -> Path:
    """
    Download and setup a dataset.
    
    Args:
        dataset_name: Name of the dataset to download
        data_dir: Directory to store the dataset
        
    Returns:
        Path to the downloaded dataset file
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(DATASETS.keys())}")
    
    dataset_info = DATASETS[dataset_name]
    output_path = data_dir / dataset_info['filename']
    
    # Check if already downloaded
    if output_path.exists():
        logger.info(f"Dataset already exists at {output_path}")
        return output_path
    
    # Download dataset
    logger.info(f"Downloading {dataset_name} dataset from {dataset_info['url']}")
    download_file(dataset_info['url'], output_path)
    logger.info(f"Downloaded {dataset_name} dataset to {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Download datasets for text generation')
    parser.add_argument('dataset', choices=list(DATASETS.keys()),
                      help='Name of the dataset to download')
    parser.add_argument('--data-dir', type=str, default='text/dataset',
                      help='Directory to store the datasets')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Convert data_dir to Path object
    data_dir = Path(args.data_dir)
    
    try:
        output_path = setup_dataset(args.dataset, data_dir)
        logger.info(f"Dataset ready at: {output_path}")
        
        # Print some statistics about the dataset
        with open(output_path, 'r', encoding='utf-8') as f:
            text = f.read()
            logger.info(f"Dataset statistics:")
            logger.info(f"- Total characters: {len(text):,}")
            logger.info(f"- Total lines: {len(text.splitlines()):,}")
            logger.info(f"- Unique characters: {len(set(text)):,}")
            
    except Exception as e:
        logger.error(f"Error setting up dataset: {e}")
        raise

if __name__ == "__main__":
    main() 