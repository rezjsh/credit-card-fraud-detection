import os
import shutil
import zipfile
from pathlib import Path
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from credit_card_fraud_detection.utils.logging_setup import logger
from credit_card_fraud_detection.components.data_ingestion.interface import IDataIngestionStrategy
from credit_card_fraud_detection.entity.config_entity import DataIngestionConfig

load_dotenv()


class LocalCSVIngestionStrategy(IDataIngestionStrategy):
    """Strategy for handling datasets supplied via local file system paths."""

    def download_data(self, config: DataIngestionConfig) -> None:
        """Stages the local source file into the ingestion staging location."""
        source_path = Path(config.source_url_or_path)

        if not source_path.exists():
            raise FileNotFoundError(f"Local source file not found at: {source_path}")

        if config.local_data_file.exists() and config.local_data_file.stat().st_size > 0:
            logger.info(f"Local file already present in staging, skipping copy: {config.local_data_file}")
            return

        logger.info(f"Staging local data file from {source_path} to {config.local_data_file}...")
        config.local_data_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, config.local_data_file)
        logger.info(f"File staged successfully at {config.local_data_file}")

    def extract_data(self, config: DataIngestionConfig) -> None:
        """
        Extracts ZIP archives into the raw landing directory, or creates zero-overhead links
        for uncompressed datasets to eliminate redundant disk footprint and I/O overhead.
        """
        if not config.local_data_file.exists():
            raise FileNotFoundError(f"Staged dataset not found at: {config.local_data_file}")

        config.unzip_dir.mkdir(parents=True, exist_ok=True)

        # 1. If the local staged file is a ZIP archive, unpack it
        if zipfile.is_zipfile(config.local_data_file):
            logger.info(f"Staged file is a ZIP archive. Unpacking {config.local_data_file.name} to {config.unzip_dir}...")
            try:
                with zipfile.ZipFile(config.local_data_file, "r") as zip_ref:
                    zip_ref.extractall(config.unzip_dir)
                logger.info(f"Successfully extracted ZIP contents into: {config.unzip_dir}")
            except zipfile.BadZipFile as e:
                raise RuntimeError(f"Corrupted ZIP file at {config.local_data_file}: {e}") from e
            return

        # 2. For uncompressed datasets (CSV, Parquet, etc.), stage via linking instead of duplicating raw bytes
        raw_target = config.unzip_dir / config.local_data_file.name

        if raw_target.exists() and raw_target.stat().st_size > 0:
            logger.info(f"Raw landing file already exists, skipping link step: {raw_target}")
            return

        logger.info(f"Staging uncompressed dataset at raw landing path: {raw_target}")

        try:
            # Tier 1: Hardlink (0 extra bytes, instantaneous 0ms execution, supported on Windows & POSIX without admin rights)
            raw_target.hardlink_to(config.local_data_file)
            logger.info(f"Hardlink created successfully (0 MB added): {raw_target} -> {config.local_data_file}")

        except OSError:
            try:
                # Tier 2: Symlink (Fallback if hardlink is restricted)
                raw_target.symlink_to(config.local_data_file.resolve())
                logger.info(f"Symlink created successfully: {raw_target} -> {config.local_data_file}")

            except OSError:
                # Tier 3: Physical Copy (Final fallback for cross-volume/mount restrictions)
                logger.warning("Hardlink/Symlink failed (cross-device filesystem restriction). Falling back to physical copy...")
                shutil.copy2(config.local_data_file, raw_target)
                logger.info(f"Fallback physical copy completed at: {raw_target}")

class KaggleAPIIngestionStrategy(IDataIngestionStrategy):
    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=1,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def download_data(self, config: DataIngestionConfig) -> None:
        """Download data from Kaggle using the Kaggle API."""
        if config.local_data_file.exists() and config.local_data_file.stat().st_size > 0:
            logger.info(f"Zip already exists, skipping download: {config.local_data_file}")
            return
        logger.info("Starting Kaggle token-based download...")

        token = os.getenv("KAGGLE_API_TOKEN")
        if not token:
            raise EnvironmentError("KAGGLE_API_TOKEN is missing")

        dataset_slug = config.source_url_or_path.strip()
        owner, name = dataset_slug.split("/", 1)
        url = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{name}"
        headers = {"Authorization": f"Bearer {token}"}

        config.local_data_file.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = config.local_data_file.with_suffix(".part")

        session = self._build_session()

        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 300), allow_redirects=True) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with open(tmp_file, "wb") as f, tqdm(
                   total=total,
                   unit="B",
                   unit_scale=True,
                   desc="Downloading Kaggle dataset"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            tmp_file.replace(config.local_data_file)
            logger.info(f"Dataset downloaded successfully to {config.local_data_file}")

        except requests.exceptions.ChunkedEncodingError as e:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)
            raise RuntimeError(
                "Kaggle download was interrupted mid-transfer. "
                "Try again, use a stable network, or switch to manual/local ingestion."
            ) from e
        except Exception:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)
            raise

    def extract_data(self, config: DataIngestionConfig) -> None:
        """Extract data from the downloaded Kaggle zip file."""
        expected_csv = config.unzip_dir / Path(config.local_data_file).name.replace(".zip", ".csv")
        if expected_csv.exists() and expected_csv.stat().st_size > 0 and zipfile.is_zipfile(config.local_data_file):
            logger.info(f"Extracted file already exists, skipping extraction: {expected_csv}")
            return
        logger.info("Extracting zip archive into raw data directory...")

        if not config.local_data_file.exists():
            raise FileNotFoundError(f"Zip file not found: {config.local_data_file}")

        config.unzip_dir.mkdir(parents=True, exist_ok=True)

        if zipfile.is_zipfile(config.local_data_file):
            logger.info(f"Payload is a valid ZIP archive. Unpacking {config.local_data_file.name}...")
            try:
                with zipfile.ZipFile(config.local_data_file, "r") as zip_ref:
                    zip_ref.extractall(config.unzip_dir)
                logger.info(f"Successfully extracted ZIP archive contents into: {config.unzip_dir}")
            except zipfile.BadZipFile as e:
                raise RuntimeError(f"Downloaded file appears to be a corrupted ZIP archive: {e}") from e
        else:
            logger.info(f"Payload is not a ZIP archive. Copying raw payload directly to landing directory...")
            target_path = config.unzip_dir / config.local_data_file.name

            if target_path.exists() and target_path.stat().st_size > 0:
                logger.info(f"Raw landing file already exists, skipping copy: {target_path}")
                return

            shutil.copy2(config.local_data_file, target_path)
            logger.info(f"Raw payload successfully staged at: {target_path}")