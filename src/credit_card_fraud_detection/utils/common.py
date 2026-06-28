import json
import os
import zipfile
import pickle
from typing import List, Dict, Any
from box import ConfigBox
import requests
import yaml
from credit_card_fraud_detection.utils.logging_setup import logger

def read_yaml_file(file_path: str) -> ConfigBox:
    """
    Read a YAML file and return its content as a ConfigBox object for easy dot-notation access.
    
    Args:
        file_path (str): Path to the YAML file.
        
    Returns:
        ConfigBox: Content of the YAML file wrapped in a ConfigBox.
    """
    try:
        logger.info(f"Reading YAML file: {file_path}")
        with open(file_path, 'r') as file:
            content = yaml.safe_load(file)
            logger.info(f"YAML file {file_path} loaded successfully")
        return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error reading YAML file {file_path}: {e}")
        raise e

def create_directory(dirs: List[str]) -> None:
    """
    Creates a list of directories if they do not already exist.
    
    Args:
        dirs (list): A list of directory paths to create.
    """
    try:
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created/Verified directory: {dir_path}")
    except Exception as e:
        logger.error(f"Error creating directory {dir_path}: {e}")
        raise e

def save_json(file_path: str, content: Dict[str, Any]) -> None:
    """
    Saves a dictionary to a JSON file. Useful for saving metrics, 
    threshold parameters, or configuration outputs.
    
    Args:
        file_path (str): Target destination path.
        content (dict): Data dictionary to be stored.
    """
    try:
        # Create parent directory if it doesn't exist
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(file_path, 'w') as file:
            json.dump(content, file, indent=4)
        logger.info(f"JSON file saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON file to {file_path}: {e}")
        raise e

def load_json(file_path: str) -> Dict[str, Any]:
    """
    Loads a JSON file and returns its content as a dictionary.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: The parsed JSON content.
    """
    try:
        logger.info(f"Loading JSON file: {file_path}")
        with open(file_path, 'r') as file:
            content = json.load(file)
        return content
    except Exception as e:
        logger.error(f"Error loading JSON file from {file_path}: {e}")
        raise e

def download_file(url: str, filename: str) -> bool:
    """
    Downloads a file from a given URL. Extremely helpful for pulling the 
    Telco Churn dataset directly from remote buckets or storage.
    
    Args:
        url (str): The URL of the file to download.
        filename (str): The local filename to save the downloaded content.
        
    Returns:
        bool: True if download is successful, False otherwise.
    """
    logger.info(f"Downloading {filename} from {url}...")
    try:
        # Create directory path if necessary
        parent_dir = os.path.dirname(filename)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Successfully downloaded {filename}.")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {filename}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred while downloading {filename}: {e}")
        return False

def extract_zip(zip_path: str, extract_to: str) -> bool:
    """
    Extracts a zip file to a specified directory.
    
    Args:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory where contents will be extracted.
        
    Returns:
        bool: True if extraction is successful, False otherwise.
    """
    logger.info(f"Extracting {zip_path} to {extract_to}...")
    try:
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Successfully extracted {zip_path}.")
        return True
    except zipfile.BadZipFile:
        logger.error(f"Error: {zip_path} is not a valid zip file.")
        return False
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        return False

def save_model(model: Any, file_path: str) -> None:
    """
    Serializes and saves a trained machine learning model or preprocessor 
    (such as a pipeline, scaler, or model strategy instance) to a file.
    
    Args:
        model (Any): The model/pipeline object to serialize.
        file_path (str): Path to save the serialized model.
    """
    try:
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.info(f"Model saved successfully to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise e

def load_model(file_path: str) -> Any:
    """
    Deserializes and loads a saved machine learning model or preprocessing pipeline.
    
    Args:
        file_path (str): Path to the saved model file.
        
    Returns:
        Any: The deserialized model or pipeline object.
    """
    try:
        logger.info(f"Loading model from: {file_path}")
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        raise e