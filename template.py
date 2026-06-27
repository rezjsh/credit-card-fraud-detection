import pathlib
import logging

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Define Project Name ---
PROJECT_SLUG = "credit_card_fraud_detection"

# --- Define Directories to Create ---
dirs_to_create = [
    f"configs",             # Centralized configuration
    f"data/raw",            # Raw data landing zone (Credit Card 2023 dataset)
    f"data/processed",      # SMOTE / Class-weighted processed outputs
    f"docs",
    f"logs",                # Execution logs
    f"notebooks",           # EDA and Threshold Optimization prototyping
    f"artifacts",           # Models, threshold curves, feature importances
    f"tests",               # Unit tests for pipelines and evaluation metrics
    
    # --- Source Code Structure (PYTHON PACKAGE) ---
    f"src/{PROJECT_SLUG}", 
    f"src/{PROJECT_SLUG}/utils",
    f"src/{PROJECT_SLUG}/components", 
    f"src/{PROJECT_SLUG}/pipeline",
    f"src/{PROJECT_SLUG}/config",    # ConfigurationManager logic
    f"src/{PROJECT_SLUG}/entity",    # config_entity.py
    f"src/{PROJECT_SLUG}/constants", # constants.py
    
    # --- Deployment App ---
    f"app",                  # Interactive Streamlit dashboard / API
]

# --- Define Files to Create ---
files_to_create = [
    # Config files
    "configs/config.yaml",
    "configs/params.yaml",
    "configs/logging_config.yaml",
    "configs/mlflow_config.yaml",   # Tracking configuration
    
    # Root files
    "README.md",
    "setup.py",
    "requirements.txt",
    ".gitignore",
    "main.py",
    "Makefile",
    "Dockerfile",
    ".env.example",
    
    # Src Package Init Files
    f"src/{PROJECT_SLUG}/__init__.py",
    f"src/{PROJECT_SLUG}/utils/__init__.py",
    f"src/{PROJECT_SLUG}/components/__init__.py",
    f"src/{PROJECT_SLUG}/pipeline/__init__.py",
    f"src/{PROJECT_SLUG}/config/__init__.py",
    f"src/{PROJECT_SLUG}/entity/__init__.py",
    f"src/{PROJECT_SLUG}/constants/__init__.py",
    
    # --- Data Ingestion Component ---
    f"src/{PROJECT_SLUG}/components/data_ingestion/__init__.py",
    f"src/{PROJECT_SLUG}/components/data_ingestion/interface.py",
    f"src/{PROJECT_SLUG}/components/data_ingestion/strategies.py",    # Local CSV vs Kaggle API download
    f"src/{PROJECT_SLUG}/components/data_ingestion/factory.py",
    f"src/{PROJECT_SLUG}/components/data_ingestion/orchestrator.py",
    
    # --- Data Validation Component ---
    f"src/{PROJECT_SLUG}/components/data_validation/__init__.py",
    f"src/{PROJECT_SLUG}/components/data_validation/interface.py",
    f"src/{PROJECT_SLUG}/components/data_validation/strategies.py",  # Checking schema, missing columns, imbalance ratios
    f"src/{PROJECT_SLUG}/components/data_validation/orchestrator.py",
    
    # --- Data Transformation Component ---
    f"src/{PROJECT_SLUG}/components/data_transformation/__init__.py",
    f"src/{PROJECT_SLUG}/components/data_transformation/interface.py",
    f"src/{PROJECT_SLUG}/components/data_transformation/strategies.py", # SMOTE vs Downsampling vs Scaling
    f"src/{PROJECT_SLUG}/components/data_transformation/orchestrator.py",
    
    # --- Model Trainer Component ---
    f"src/{PROJECT_SLUG}/components/model_trainer/__init__.py",
    f"src/{PROJECT_SLUG}/components/model_trainer/interface.py",
    f"src/{PROJECT_SLUG}/components/model_trainer/strategies.py",   # LogisticRegression vs RandomForest vs XGBoost (with class weights)
    f"src/{PROJECT_SLUG}/components/model_trainer/factory.py",
    f"src/{PROJECT_SLUG}/components/model_trainer/orchestrator.py",
    
    # --- Model Evaluation Component (Customized for Imbalance) ---
    f"src/{PROJECT_SLUG}/components/model_evaluation/__init__.py",
    f"src/{PROJECT_SLUG}/components/model_evaluation/metrics.py",     # Precision, Recall, F1, PR-AUC, False Positives/Negatives cost tracking
    f"src/{PROJECT_SLUG}/components/model_evaluation/threshold_optimizer.py", # Optimizing decision threshold to reduce FN cost
    
    # Pipeline Files
    f"src/{PROJECT_SLUG}/pipeline/stage_01_data_ingestion.py",
    f"src/{PROJECT_SLUG}/pipeline/stage_02_data_validation.py",
    f"src/{PROJECT_SLUG}/pipeline/stage_03_data_transformation.py",
    f"src/{PROJECT_SLUG}/pipeline/stage_04_model_trainer.py",
    f"src/{PROJECT_SLUG}/pipeline/stage_05_model_evaluation.py",
    
    # Config/Entity/Constants Files
    f"src/{PROJECT_SLUG}/config/configuration.py",
    f"src/{PROJECT_SLUG}/entity/config_entity.py",
    f"src/{PROJECT_SLUG}/constants/constants.py",
    
    # Utility Files
    f"src/{PROJECT_SLUG}/utils/common.py", 
    
    # Interactive Frontend / Demo
    "app/app.py",                    # Streamlit Application code
    "app/explainability.py",         # SHAP or LIME interpretation utilities
]

# --- Basic Gitignore Content ---
gitignore_content = """
# Standard Python ignores
__pycache__/
*.py[cod]
*.so

# Environments
.env
.venv
env/
venv/
environment.yml

# Data and Logs
data/
logs/
*.log

# Models/Checkpoints / MLflow Local DB
checkpoints/
models/
mlruns/
.mlflow/

# IDE files
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints

# models, plots, and other artifacts
artifacts/
"""

# --- Create Structure ---
logging.info(f"Starting project structure creation for package: {PROJECT_SLUG}")

# Create directories
for dir_path_str in dirs_to_create:
    path = pathlib.Path(dir_path_str)
    try:
        path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory (or verified exists): {path}")
    except OSError as e:
        logging.error(f"Failed to create directory {path}: {e}")

# Create files
for file_path_str in files_to_create:
    file_path = pathlib.Path(file_path_str)
    
    # Ensure parent directory exists before touching the file
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create file if it doesn't exist
    if not file_path.exists():
        try:
            file_path.touch()
            logging.info(f"Created file: {file_path}")
        except Exception as e:
            logging.error(f"Failed to create file {file_path}: {e}")
            continue

        # Add initial content to __init__.py files
        if file_path.name == "__init__.py":
            logging.info(f"Initialized package: {file_path}")
            
        # Add initial content to README.md
        if file_path.name == "README.md" and file_path.stat().st_size == 0:
            markdown_title = f"# {PROJECT_SLUG.replace('_', ' ').title()}\n\n"
            markdown_body = (
                "## Project Highlights\n"
                "- **Architecture**: Production-grade Strategy & Factory patterns across MLOps stages.\n"
                "- **Imbalance Management**: SMOTE vs. Class-Weights comparison tracked via **MLflow**.\n"
                "- **Evaluation Focus**: PR-AUC optimization over accuracy to prioritize cost-driven False Negative minimization.\n"
                "- **Deployment**: Streamlit Dashboard with local model explainability.\n"
            )
            file_path.write_text(markdown_title + markdown_body, encoding='utf-8')
            logging.info(f"Added tailored title and highlights to {file_path.name}")
            
# Create or update .gitignore in the root directory
gitignore_path = pathlib.Path(".gitignore")
try:
    existing_lines = set()
    if gitignore_path.exists():
        with open(gitignore_path, "r", encoding='utf-8') as f:
            existing_lines = set(line.strip() for line in f.read().splitlines() if line.strip())

    new_lines = [line.strip() for line in gitignore_content.strip().splitlines() if line.strip()]
    
    lines_to_add = [line for line in new_lines if line not in existing_lines]
    
    if lines_to_add:
        with open(gitignore_path, "a", encoding='utf-8') as f:
            f.write("\n") # Ensure separation from previous content
            f.write("\n".join(lines_to_add))
        logging.info(f"Updated .gitignore with {len(lines_to_add)} new lines.")
    else:
        logging.info(".gitignore file is up-to-date.")

except Exception as e:
    logging.error(f"Failed to handle .gitignore: {e}")

logging.info("Project structure creation process finished successfully.")